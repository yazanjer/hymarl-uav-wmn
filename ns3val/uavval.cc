// Packet-level ns-3 validation of the UAV-WMN controllers (evaluation only).
// The trained actors (exported from MATLAB) and the rule-based components act on a
// UAV mesh simulated by ns-3 (802.11n, 2.4 GHz, SpectrumWifiPhy, log-distance path
// loss with log-normal per-packet shadowing). User positions and demands replay the
// MATLAB evaluation episodes exactly. One process = one (method, seed, episode).
//
//   ./ns3 run "uavval --policy=P.txt --trace=T.bin --out=O.csv [--mode=ns3|model]"
//
// mode=model runs only the analytical network model (C++ port of NetworkSim.m) and is
// used to check the port against the MATLAB results.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

using namespace ns3;

// ------------------------------------------------------------------ configuration (Scenario A)
struct Cfg
{
    int N = 10, U = 50, K = 1, T = 60;
    double area = 1000, z = 100, dt = 0.1, vmax = 20;
    double PLd0 = 40, d0 = 1, plExp = 2.5, shadow = 5.0;
    double Pmax = 20, Pmin = 0, sens = -85, N0dBm = -95, BW = 20e6, kappa = 0.30, GammaMin = 3.0;
    int CWmin = 16, mB = 6;
    double slot = 9e-6, SIFS = 16e-6, DIFS = 34e-6, Tpacket = 512 * 8, header = 40 * 8, c0 = 3e8;
    double Dmax = 2000;
};
static Cfg C;

// ------------------------------------------------------------------ policy
struct Layer
{
    int rows, cols;
    std::vector<double> W, b; // W row-major rows x cols
};
struct Policy
{
    std::string method, assign, chanMode, powerMode;
    int dObs = 0, actDim = 0, contDim = 0, chanN = 0, powN = 0, K = 1;
    bool aug = false, useLyap = false, learned = true;
    double kL = 0.5, Ttarget = 0.3, dPmax = 1.0, dS = 0.2;
    std::vector<Layer> layers;
};

static Policy
LoadPolicy(const std::string& fn)
{
    Policy p;
    std::ifstream f(fn);
    NS_ABORT_MSG_IF(!f, "cannot open policy " << fn);
    int nl, aug, lyap, learned;
    f >> p.method >> p.assign >> p.chanMode >> p.powerMode >> p.dObs >> p.actDim >> p.contDim >> p.chanN >>
        p.powN >> p.K >> aug >> lyap >> learned >> p.kL >> p.Ttarget >> p.dPmax >> p.dS >> nl;
    p.aug = aug;
    p.useLyap = lyap;
    p.learned = learned;
    for (int l = 0; l < nl; ++l)
    {
        Layer L;
        f >> L.rows >> L.cols;
        L.W.resize(L.rows * L.cols);
        L.b.resize(L.rows);
        for (auto& w : L.W)
            f >> w;
        for (auto& b : L.b)
            f >> b;
        p.layers.push_back(L);
    }
    return p;
}

static std::vector<double>
Forward(const Policy& p, const std::vector<double>& x)
{
    std::vector<double> h = x;
    for (size_t l = 0; l < p.layers.size(); ++l)
    {
        const Layer& L = p.layers[l];
        std::vector<double> y(L.rows);
        for (int r = 0; r < L.rows; ++r)
        {
            double s = L.b[r];
            for (int c = 0; c < L.cols; ++c)
                s += L.W[r * L.cols + c] * h[c];
            y[r] = (l + 1 < p.layers.size()) ? std::max(s, 0.0) : s;
        }
        h.swap(y);
    }
    return h;
}

// ------------------------------------------------------------------ trace
struct Trace
{
    int U, T;
    std::vector<double> demand;
    std::vector<std::vector<double>> x, y; // [t][u]
};

static Trace
LoadTrace(const std::string& fn)
{
    Trace tr;
    std::ifstream f(fn, std::ios::binary);
    NS_ABORT_MSG_IF(!f, "cannot open trace " << fn);
    int32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    tr.U = hdr[0];
    tr.T = hdr[1];
    tr.demand.resize(tr.U);
    f.read(reinterpret_cast<char*>(tr.demand.data()), tr.U * sizeof(double));
    for (int t = 0; t <= tr.T; ++t)
    {
        std::vector<double> buf(2 * tr.U);
        f.read(reinterpret_cast<char*>(buf.data()), 2 * tr.U * sizeof(double));
        tr.x.emplace_back(buf.begin(), buf.begin() + tr.U);
        tr.y.emplace_back(buf.begin() + tr.U, buf.end());
    }
    return tr;
}

// ------------------------------------------------------------------ analytical model (port of NetworkSim.m)
struct State
{
    std::vector<double> px, py, vx, vy, P; // UAVs
    std::vector<int> ch;                   // 1..K
    std::vector<int> assign, prevAssign;   // 1..N (0 = none)
    std::vector<double> ux, uy, dem;       // users
    double gx[2], gy[2], gz[2];
};

struct Net
{
    std::vector<std::vector<double>> PrxUU, Guu, Gug, PrxUG; // PrxUU[rx][tx]
    std::vector<std::vector<double>> XsUU, XsUG;            // shadowing draws XsUU[rx][tx], XsUG[uav][gw]
    std::vector<std::vector<bool>> adj;
    std::vector<double> hop, sinr, rate, Dmac, load, rho, e2e, deliv;
    std::vector<int> nextHop, destGw; // nextHop: UAV index, or -1-g for gateway g; -100 none
};

static std::mt19937_64 gRng(1);
static std::normal_distribution<double> gN01(0.0, 1.0);

static double
Overlap(int a, int b)
{
    if (a == b)
        return 1.0;
    if (std::abs(a - b) == 1)
        return C.kappa;
    return 0.0;
}

static double
MacDelay(int n)
{
    n = std::max(1, n);
    double W = C.CWmin;
    int m = C.mB;
    double tau = 0.02, p = 0;
    for (int it = 0; it < 200; ++it)
    {
        p = 1 - std::pow(1 - tau, n - 1);
        double p2m = std::pow(2 * p, m);
        double denom = (1 - 2 * p) * (W + 1) + p * W * (1 - p2m);
        double tn = 2 * (1 - 2 * p) / std::max(denom, 1e-9);
        tn = std::min(std::max(tn, 1e-6), 0.5);
        if (std::abs(tn - tau) < 1e-9)
        {
            tau = tn;
            break;
        }
        tau = 0.5 * tau + 0.5 * tn;
    }
    double Tframe = (C.header + C.Tpacket) / C.BW;
    double Ts = C.DIFS + Tframe + C.SIFS, Tc = C.DIFS + Tframe;
    double Ptr = 1 - std::pow(1 - tau, n);
    double Ps = n * tau * std::pow(1 - tau, n - 1) / std::max(Ptr, 1e-9);
    double Eslot = (1 - Ptr) * C.slot + Ptr * Ps * Ts + Ptr * (1 - Ps) * Tc;
    double avgW = 0;
    for (int k = 0; k < m; ++k)
        avgW += ((std::pow(2, k) * W - 1) / 2) * std::pow(p, k);
    avgW = avgW * (1 - p) / std::max(1 - std::pow(p, m), 1e-9);
    return avgW * Eslot + Ts;
}

static void
PathGain(double dx, double dy, double dz, double Pt, double& prx, double& g, double& Xs)
{
    double d = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), C.d0);
    double PL = C.PLd0 + 10 * C.plExp * std::log10(d / C.d0);
    Xs = C.shadow * gN01(gRng);
    prx = Pt - PL - Xs;
    g = std::pow(10.0, (-PL - Xs) / 10);
}

static Net
ComputeNetwork(const State& s)
{
    int N = C.N;
    Net n;
    n.PrxUU.assign(N, std::vector<double>(N, -1e9));
    n.Guu.assign(N, std::vector<double>(N, 0));
    n.XsUU.assign(N, std::vector<double>(N, 0));
    n.XsUG.assign(N, std::vector<double>(2, 0));
    for (int i = 0; i < N; ++i)     // transmitter i
        for (int r = 0; r < N; ++r) // receiver r
        {
            double prx, g, xs;
            PathGain(s.px[i] - s.px[r], s.py[i] - s.py[r], 0, s.P[i], prx, g, xs);
            if (r != i)
            {
                n.PrxUU[r][i] = prx;
                n.Guu[r][i] = g;
                n.XsUU[r][i] = xs;
            }
        }
    n.adj.assign(N, std::vector<bool>(N, false));
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            if (a != b && (n.PrxUU[a][b] >= C.sens || n.PrxUU[b][a] >= C.sens))
                n.adj[a][b] = true;
    n.Gug.assign(N, std::vector<double>(2, 0));
    n.PrxUG.assign(N, std::vector<double>(2, -1e9));
    for (int gw = 0; gw < 2; ++gw)
        for (int i = 0; i < N; ++i)
        {
            double prx, g, xs;
            PathGain(s.px[i] - s.gx[gw], s.py[i] - s.gy[gw], C.z - s.gz[gw], s.P[i], prx, g, xs);
            n.PrxUG[i][gw] = prx;
            n.Gug[i][gw] = g;
            n.XsUG[i][gw] = xs;
        }
    // BFS hop count
    const double INF = std::numeric_limits<double>::infinity();
    n.hop.assign(N, INF);
    std::vector<int> frontier;
    for (int i = 0; i < N; ++i)
        if (n.PrxUG[i][0] >= C.sens || n.PrxUG[i][1] >= C.sens)
        {
            n.hop[i] = 1;
            frontier.push_back(i);
        }
    int d = 1;
    while (!frontier.empty())
    {
        ++d;
        std::vector<int> nxt;
        for (int f : frontier)
            for (int j = 0; j < N; ++j)
                if (n.adj[f][j] && !std::isfinite(n.hop[j]))
                {
                    n.hop[j] = d;
                    nxt.push_back(j);
                }
        frontier = nxt;
    }
    // next hop and SINR
    n.sinr.assign(N, 0);
    n.rate.assign(N, 0);
    n.nextHop.assign(N, -100);
    double N0 = std::pow(10.0, (C.N0dBm - 30) / 10);
    std::vector<double> Plin(N);
    for (int i = 0; i < N; ++i)
        Plin[i] = std::pow(10.0, (s.P[i] - 30) / 10);
    for (int i = 0; i < N; ++i)
    {
        int best = -1;
        double bg = -1;
        for (int j = 0; j < N; ++j)
            if (n.adj[i][j] && n.hop[j] < n.hop[i] && n.Guu[i][j] > bg)
            {
                bg = n.Guu[i][j];
                best = j;
            }
        double sig;
        if (best < 0)
        {
            int gb = n.Gug[i][0] >= n.Gug[i][1] ? 0 : 1;
            sig = Plin[i] * n.Gug[i][gb];
            if (std::isfinite(n.hop[i]))
                n.nextHop[i] = -1 - gb;
        }
        else
        {
            sig = Plin[i] * n.Guu[i][best];
            if (std::isfinite(n.hop[i]))
                n.nextHop[i] = best;
        }
        double interf = 0;
        for (int k = 0; k < N; ++k)
        {
            if (k == i || n.PrxUU[i][k] < C.sens - 10)
                continue;
            double kap = Overlap(s.ch[k], s.ch[i]);
            if (kap > 0)
                interf += Plin[k] * n.Guu[i][k] * kap;
        }
        n.sinr[i] = sig / (N0 + interf);
        n.rate[i] = C.BW * std::log2(1 + std::max(n.sinr[i], 0.0)) / 1e6;
    }
    // destination gateway along the next-hop chain
    n.destGw.assign(N, -1);
    for (int i = 0; i < N; ++i)
    {
        int cur = i, guard = 0;
        while (cur >= 0 && guard++ < 2 * N)
            cur = n.nextHop[cur];
        if (cur <= -1 && cur >= -2)
            n.destGw[i] = -1 - cur;
    }
    n.Dmac.assign(N, 0);
    for (int i = 0; i < N; ++i)
    {
        int nc = 0;
        for (int k = 0; k < N; ++k)
            nc += (s.ch[k] == s.ch[i]);
        n.Dmac[i] = MacDelay(nc);
    }
    n.load.assign(N, 0);
    for (int u = 0; u < C.U; ++u)
        if (s.assign[u] > 0)
            n.load[s.assign[u] - 1] += s.dem[u];
    n.rho.assign(N, 0);
    n.e2e.assign(N, 0);
    n.deliv.assign(N, 0);
    for (int i = 0; i < N; ++i)
    {
        double mu = std::max(n.rate[i], 1e-3) * 1e6 / C.Tpacket;
        double lam = n.load[i] * 1e6 / C.Tpacket;
        n.rho[i] = std::min(lam / mu, 0.98);
        double Wq = 1.0 / std::max(mu * (1 - n.rho[i]), 1e-3);
        double prop = std::sqrt(std::pow(s.px[i] - s.gx[0], 2) + std::pow(s.py[i] - s.gy[0], 2) +
                                std::pow(C.z - s.gz[0], 2)) /
                      C.c0;
        bool conn = std::isfinite(n.hop[i]);
        double hopEff = conn ? n.hop[i] : 8;
        double e = (n.Dmac[i] + Wq + prop) * std::max(hopEff, 1.0) * 1000;
        if (!conn)
            e = C.Dmax;
        n.e2e[i] = std::min(e, C.Dmax);
        double sinrOK = std::min(std::max(n.sinr[i] / std::pow(10.0, C.GammaMin / 10), 0.0), 1.0);
        double served = std::min(1.0, mu / std::max(lam, 1e-6));
        n.deliv[i] = std::min(std::max((conn ? 1.0 : 0.0) * sinrOK * served, 0.0), 1.0);
    }
    return n;
}

// ------------------------------------------------------------------ rules
static std::vector<int>
LoadBalanceAssign(const State& s)
{
    std::vector<int> order(C.U);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s.dem[a] > s.dem[b]; });
    std::vector<double> load(C.N, 0);
    std::vector<int> a(C.U, 0);
    double reach = 0.30 * C.area;
    for (int u : order)
    {
        std::vector<int> cand;
        int nearest = 0;
        double dn = 1e18;
        for (int j = 0; j < C.N; ++j)
        {
            double d = std::hypot(s.ux[u] - s.px[j], s.uy[u] - s.py[j]);
            if (d < reach)
                cand.push_back(j);
            if (d < dn)
            {
                dn = d;
                nearest = j;
            }
        }
        if (cand.empty())
            cand.push_back(nearest);
        int j = cand[0];
        for (int c : cand)
            if (load[c] < load[j])
                j = c;
        a[u] = j + 1;
        load[j] += s.dem[u];
    }
    return a;
}

static std::vector<int>
NearestRouter(const State& s)
{
    std::vector<int> a(C.U);
    for (int u = 0; u < C.U; ++u)
    {
        int best = 0;
        double bd = 1e18;
        for (int j = 0; j < C.N; ++j)
        {
            double d = std::hypot(s.ux[u] - s.px[j], s.uy[u] - s.py[j]);
            if (d < bd)
            {
                bd = d;
                best = j;
            }
        }
        a[u] = best + 1;
    }
    return a;
}

static std::vector<int>
GreedyChannels(const State& s, const Net& n, int K)
{
    int N = C.N;
    std::vector<std::vector<double>> W(N, std::vector<double>(N, 0));
    std::vector<int> deg(N, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double a = (i != j) ? std::pow(10.0, (n.PrxUU[i][j] - 30) / 10) : 0;
            double b = (i != j) ? std::pow(10.0, (n.PrxUU[j][i] - 30) / 10) : 0;
            W[i][j] = a + b;
            bool inter = (i != j) && (n.PrxUU[i][j] >= C.sens - 10 || n.PrxUU[j][i] >= C.sens - 10);
            deg[i] += inter;
        }
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return deg[a] > deg[b]; });
    std::vector<int> ch(N, 0);
    for (int i : order)
    {
        int bk = 1;
        double bc = 1e300;
        for (int k = 1; k <= K; ++k)
        {
            double cost = 0;
            for (int j = 0; j < N; ++j)
                if (ch[j] > 0)
                    cost += W[i][j] * Overlap(k, ch[j]);
            if (cost < bc)
            {
                bc = cost;
                bk = k;
            }
        }
        ch[i] = bk;
    }
    return ch;
}

// ------------------------------------------------------------------ observations
static std::vector<std::vector<double>>
GetObs(const State& s, const Net& n, const std::vector<double>& e2eObs, int K, bool aug)
{
    int N = C.N;
    std::vector<std::vector<double>> o(N);
    std::vector<double> occ(K + 1, 0);
    for (int i = 0; i < N; ++i)
        occ[s.ch[i]] += 1.0 / N;
    for (int i = 0; i < N; ++i)
    {
        double gx = s.gx[0] - s.px[i], gy = s.gy[0] - s.py[i];
        double nr = std::max(std::hypot(gx, gy), 1e-6);
        int degN = 0;
        for (int j = 0; j < N; ++j)
            degN += n.adj[i][j];
        std::vector<double> v = {s.px[i] / C.area,
                                 s.py[i] / C.area,
                                 s.vx[i] / C.vmax,
                                 s.vy[i] / C.vmax,
                                 (s.P[i] - C.Pmin) / (C.Pmax - C.Pmin),
                                 std::min(n.rho[i], 1.0),
                                 std::tanh(e2eObs[i] / 100),
                                 gx / nr,
                                 gy / nr,
                                 std::min(double(degN) / N, 1.0),
                                 std::tanh(10 * std::log10(std::max(n.sinr[i], 1e-3)) / 20),
                                 std::min(n.load[i] / 10, 1.0)};
        for (int k = 1; k <= K; ++k)
            v.push_back(occ[k]);
        o[i] = v;
    }
    if (!aug)
        return o;
    std::vector<std::vector<double>> oa(N);
    for (int i = 0; i < N; ++i)
    {
        double ws = 0;
        for (int j = 0; j < N; ++j)
            if (n.adj[i][j])
                ws += n.Guu[i][j];
        std::vector<double> agg(o[i].size(), 0);
        for (int j = 0; j < N; ++j)
            if (n.adj[i][j])
            {
                double w = n.Guu[i][j] / (ws + 1e-9);
                for (size_t k = 0; k < agg.size(); ++k)
                    agg[k] += w * o[j][k];
            }
        oa[i] = o[i];
        oa[i].insert(oa[i].end(), agg.begin(), agg.end());
    }
    return oa;
}

// ------------------------------------------------------------------ packet-level network
// Shadowing held constant within a control step and equal to the draws of the network
// model of that step, so that routing and the physical channel see the same realisation.
static std::vector<std::vector<double>> gShadow; // [txNode][rxNode] dB

class StepShadowLoss : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("StepShadowLoss").SetParent<PropagationLossModel>().AddConstructor<StepShadowLoss>();
        return tid;
    }

  private:
    double DoCalcRxPower(double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
    {
        uint32_t ta = a->GetObject<Node>()->GetId(), rb = b->GetObject<Node>()->GetId();
        if (ta < gShadow.size() && rb < gShadow.size())
            return txPowerDbm - gShadow[ta][rb];
        return txPowerDbm;
    }
    int64_t DoAssignStreams(int64_t) override { return 0; }
};

static void
UpdateShadow(const Net& n)
{
    int N = C.N;
    gShadow.assign(N + 2, std::vector<double>(N + 2, 0));
    for (int r = 0; r < N; ++r)
        for (int t = 0; t < N; ++t)
            gShadow[t][r] = n.XsUU[r][t];
    for (int i = 0; i < N; ++i)
        for (int g = 0; g < 2; ++g)
        {
            gShadow[i][N + g] = n.XsUG[i][g];
            gShadow[N + g][i] = n.XsUG[i][g];
        }
}

static const uint16_t kPort = 9;

class TagHeader : public Header
{
  public:
    uint32_t uav = 0, step = 0;
    int64_t ts = 0;
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("TagHeader").SetParent<Header>().AddConstructor<TagHeader>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 16; }
    void Serialize(Buffer::Iterator it) const override
    {
        it.WriteHtonU32(uav);
        it.WriteHtonU32(step);
        it.WriteHtonU64(uint64_t(ts));
    }
    uint32_t Deserialize(Buffer::Iterator it) override
    {
        uav = it.ReadNtohU32();
        step = it.ReadNtohU32();
        ts = int64_t(it.ReadNtohU64());
        return 16;
    }
    void Print(std::ostream& os) const override { os << uav; }
};

struct Meas
{
    std::vector<std::vector<double>> sent, recv, dsum; // [step][uav], by send step
    std::vector<double> winRecv, winDsum;              // receive-time window of current step
};
static Meas M;
static int gStep = 0;
static NodeContainer gUav, gGw;
static std::vector<Ptr<Socket>> gTx;
static std::vector<double> gRateMbps;
static std::vector<int> gDest; // gateway index per UAV (-1 none)
static std::vector<std::vector<Ipv4Address>> gAddr; // [node][channel-1]; node index: UAVs then GWs
static int gK = 1;
static bool gGenerating = true;

static void
RecvCb(Ptr<Socket> sock)
{
    Ptr<Packet> p;
    while ((p = sock->Recv()))
    {
        TagHeader h;
        p->RemoveHeader(h);
        double delay = (Simulator::Now().GetNanoSeconds() - h.ts) * 1e-6; // ms
        if (delay > C.Dmax || h.step >= M.sent.size())
            continue;
        M.recv[h.step][h.uav] += 1;
        M.dsum[h.step][h.uav] += delay;
        M.winRecv[h.uav] += 1;
        M.winDsum[h.uav] += delay;
    }
}

static void
SendPacket(int i)
{
    if (!gGenerating)
        return;
    double r = gRateMbps[i];
    if (r > 1e-6)
    {
        M.sent[gStep][i] += 1;
        if (gDest[i] >= 0)
        {
            TagHeader h;
            h.uav = i;
            h.step = gStep;
            h.ts = Simulator::Now().GetNanoSeconds();
            Ptr<Packet> p = Create<Packet>(512 - 16);
            p->AddHeader(h);
            gTx[i]->SendTo(p, 0, InetSocketAddress(gAddr[C.N + gDest[i]][0], kPort));
        }
        double gap = C.Tpacket / (r * 1e6);
        Simulator::Schedule(Seconds(gap), &SendPacket, i);
    }
    else
        Simulator::Schedule(Seconds(C.dt / 10), &SendPacket, i);
}

static void
InstallRoutes(const Net& n, const State& s)
{
    Ipv4StaticRoutingHelper h;
    for (int i = 0; i < C.N; ++i)
    {
        Ptr<Ipv4> ip = gUav.Get(i)->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> rt = h.GetStaticRouting(ip);
        while (rt->GetNRoutes() > 0)
            rt->RemoveRoute(0);
        // connected subnets
        for (int k = 0; k < gK; ++k)
        {
            uint32_t ifc = k + 1;
            rt->AddNetworkRouteTo(Ipv4Address(("10." + std::to_string(k + 1) + ".0.0").c_str()),
                                  Ipv4Mask("255.255.0.0"),
                                  ifc);
        }
        int nh = n.nextHop[i];
        int dst = n.destGw[i];
        if (nh == -100 || dst < 0)
            continue;
        int c = (gK == 1) ? 1 : s.ch[i];
        Ipv4Address nhAddr = (nh >= 0) ? gAddr[nh][c - 1] : gAddr[C.N + (-1 - nh)][c - 1];
        rt->AddHostRouteTo(gAddr[C.N + dst][0], nhAddr, c);
    }
}

static void
SetTxPower(int i, double P)
{
    Ptr<Node> node = gUav.Get(i);
    for (uint32_t d = 0; d < node->GetNDevices(); ++d)
    {
        Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(node->GetDevice(d));
        if (!wd)
            continue;
        wd->GetPhy()->SetTxPowerStart(P);
        wd->GetPhy()->SetTxPowerEnd(P);
    }
}

// ------------------------------------------------------------------ main loop
struct Ctx
{
    Policy pol;
    Trace tr;
    State s;
    Net net;
    bool ns3mode;
    std::vector<std::vector<double>> rows; // per-step metrics
    std::vector<double> e2eObs;
};
static Ctx X;

static void
Act(std::vector<double>& dvx, std::vector<double>& dvy, std::vector<double>& dP, std::vector<int>& ch,
    std::vector<int>& asg)
{
    const Policy& p = X.pol;
    int N = C.N, K = p.K;
    dvx.assign(N, 0);
    dvy.assign(N, 0);
    dP.assign(N, 0);
    ch = X.s.ch;
    if (p.method == "HeurMC")
    {
        ch = GreedyChannels(X.s, X.net, K);
        asg = LoadBalanceAssign(X.s);
        return;
    }
    auto obs = GetObs(X.s, X.net, X.e2eObs, K, p.aug);
    std::vector<int> greedy;
    if (p.chanN == 0 && p.chanMode == "greedy" && K > 1)
        greedy = GreedyChannels(X.s, X.net, K);
    for (int i = 0; i < N; ++i)
    {
        auto y = Forward(p, obs[i]);
        std::vector<double> cont(p.contDim);
        for (int k = 0; k < p.contDim; ++k)
            cont[k] = std::max(std::min(std::tanh(y[k]), 1.0), -1.0);
        int off = p.contDim;
        if (p.chanN > 0)
        {
            int best = 0;
            for (int k = 1; k < p.chanN; ++k)
                if (y[off + k] > y[off + best])
                    best = k;
            ch[i] = best + 1;
            off += p.chanN;
        }
        else if (!greedy.empty())
            ch[i] = greedy[i];
        else
            ch[i] = (i % K) + 1;
        double d;
        if (p.powN > 0)
        {
            int best = 0;
            for (int k = 1; k < p.powN; ++k)
                if (y[off + k] > y[off + best])
                    best = k;
            d = (best - 1) * p.dS;
        }
        else
            d = cont[2] * p.dPmax;
        if (p.useLyap)
            d += p.kL * std::max(std::tanh(X.e2eObs[i] / 100) - p.Ttarget, 0.0);
        dvx[i] = cont[0] * 2;
        dvy[i] = cont[1] * 2;
        dP[i] = d;
    }
    if (p.assign == "L")
        asg = LoadBalanceAssign(X.s);
    else
        asg.clear();
}

static void
RecordStep(int t)
{
    // metrics of step t (traffic sent during step t), plus model metrics on the same state
    int N = C.N;
    std::vector<double> row;
    const Net& n = X.net;
    // model metrics
    double mDelay = 0, mPdr = 0, mThr = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < N; ++i)
    {
        mDelay += n.e2e[i] / N;
        mPdr += n.deliv[i] / N;
        double th = n.deliv[i] * n.load[i];
        mThr += th;
        s1 += th;
        s2 += th * th;
    }
    double mFair = s1 * s1 / (N * s2 + 1e-9);
    double lm = 0, lv = 0;
    for (double l : n.load)
        lm += l / N;
    for (double l : n.load)
        lv += (l - lm) * (l - lm) / (N - 1);
    int ho = 0;
    for (int u = 0; u < C.U; ++u)
        ho += (X.s.assign[u] != X.s.prevAssign[u]);
    double hor = double(ho) / C.U;
    row = {double(t), mDelay, mPdr, mThr, mFair, lv, hor};
    X.rows.push_back(row);
}

static void
StepEvent(int t)
{
    gStep = t;
    // observation delay: measured over the previous step (ns3 mode), model otherwise
    if (X.ns3mode && t > 0)
        for (int i = 0; i < C.N; ++i)
        {
            if (M.winRecv[i] > 0)
                X.e2eObs[i] = M.winDsum[i] / M.winRecv[i];
            else if (X.net.load[i] > 1e-9)
                X.e2eObs[i] = C.Dmax;
            else
                X.e2eObs[i] = X.net.e2e[i];
        }
    else
        X.e2eObs = X.net.e2e;
    std::fill(M.winRecv.begin(), M.winRecv.end(), 0);
    std::fill(M.winDsum.begin(), M.winDsum.end(), 0);

    std::vector<double> dvx, dvy, dP;
    std::vector<int> ch, asg;
    Act(dvx, dvy, dP, ch, asg);
    State& s = X.s;
    for (int i = 0; i < C.N; ++i)
    {
        s.vx[i] += dvx[i];
        s.vy[i] += dvy[i];
        double sp = std::hypot(s.vx[i], s.vy[i]);
        if (sp > C.vmax)
        {
            s.vx[i] *= C.vmax / sp;
            s.vy[i] *= C.vmax / sp;
        }
        s.px[i] = std::min(std::max(s.px[i] + s.vx[i] * C.dt, 0.0), C.area);
        s.py[i] = std::min(std::max(s.py[i] + s.vy[i] * C.dt, 0.0), C.area);
        s.P[i] = std::min(std::max(s.P[i] + dP[i], C.Pmin), C.Pmax);
        s.ch[i] = std::min(std::max(ch[i], 1), X.pol.K);
    }
    s.prevAssign = s.assign;
    s.assign = asg.empty() ? NearestRouter(s) : asg;
    for (int u = 0; u < C.U; ++u)
    {
        s.ux[u] = X.tr.x[t + 1][u];
        s.uy[u] = X.tr.y[t + 1][u];
    }
    X.net = ComputeNetwork(s);
    RecordStep(t);
    if (X.ns3mode)
    {
        for (int i = 0; i < C.N; ++i)
        {
            gUav.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(s.px[i], s.py[i], C.z));
            SetTxPower(i, s.P[i]);
            gRateMbps[i] = X.net.load[i];
            gDest[i] = X.net.destGw[i];
        }
        InstallRoutes(X.net, s);
        UpdateShadow(X.net);
        if (t + 1 < C.T)
            Simulator::Schedule(Seconds(C.dt), &StepEvent, t + 1);
        else
            Simulator::Schedule(Seconds(C.dt), [] { gGenerating = false; });
    }
}

int
main(int argc, char* argv[])
{
    std::string polFn, trFn, outFn = "out.csv", mode = "ns3", method = "";
    uint32_t seed = 1;
    double loadScale = 1.0;
    std::string scen = "A";
    int nUav = 10;
    CommandLine cmd;
    cmd.AddValue("policy", "policy file (or HeurMC)", polFn);
    cmd.AddValue("trace", "trace file", trFn);
    cmd.AddValue("out", "output csv", outFn);
    cmd.AddValue("mode", "ns3 or model", mode);
    cmd.AddValue("seed", "rng seed", seed);
    cmd.AddValue("loadScale", "multiplier of user demand", loadScale);
    cmd.AddValue("scenario", "A, B or C", scen);
    cmd.AddValue("N", "number of UAVs", nUav);
    cmd.Parse(argc, argv);
    C.N = nUav;
    if (scen == "B")
    {
        C.area = 500;
        C.plExp = 2.5;
        C.shadow = 3.0;
    }
    else if (scen == "C")
    {
        C.area = 2000;
        C.plExp = 2.2;
        C.shadow = 2.0;
    }

    if (polFn == "HeurMC")
    {
        X.pol.method = "HeurMC";
        X.pol.K = 3;
        X.pol.assign = "L";
    }
    else
        X.pol = LoadPolicy(polFn);
    C.K = X.pol.K;
    gK = C.K;
    X.tr = LoadTrace(trFn);
    C.U = X.tr.U;
    C.T = X.tr.T;
    X.ns3mode = (mode == "ns3");
    gRng.seed(seed);
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);

    // initial state (NetworkSim.reset)
    State& s = X.s;
    int g = int(std::ceil(std::sqrt(double(C.N))));
    std::vector<double> lx, ly;
    for (int a = 0; a < g; ++a)
        lx.push_back(0.15 + (0.85 - 0.15) * a / (g - 1));
    for (int a = 0; a < g; ++a)
        ly.push_back(0.05 + (0.95 - 0.05) * a / (g - 1));
    // meshgrid column-major: (:) runs over y first within each x column
    for (int ix = 0; ix < g && int(s.px.size()) < C.N; ++ix)
        for (int iy = 0; iy < g && int(s.px.size()) < C.N; ++iy)
        {
            s.px.push_back(lx[ix] * C.area);
            s.py.push_back(ly[iy] * C.area);
        }
    s.vx.assign(C.N, 0);
    s.vy.assign(C.N, 0);
    s.P.assign(C.N, C.Pmax - 6);
    s.ch.resize(C.N);
    for (int i = 0; i < C.N; ++i)
        s.ch[i] = (i % C.K) + 1;
    s.assign.assign(C.U, 0);
    s.prevAssign.assign(C.U, 0);
    s.ux = X.tr.x[0];
    s.uy = X.tr.y[0];
    s.dem = X.tr.demand;
    for (auto& v : s.dem)
        v *= loadScale;
    s.gx[0] = C.area * 0.5;
    s.gy[0] = 0;
    s.gz[0] = 0;
    s.gx[1] = C.area * 0.5;
    s.gy[1] = C.area;
    s.gz[1] = 0;
    X.net = ComputeNetwork(s);
    X.e2eObs = X.net.e2e;

    M.sent.assign(C.T, std::vector<double>(C.N, 0));
    M.recv.assign(C.T, std::vector<double>(C.N, 0));
    M.dsum.assign(C.T, std::vector<double>(C.N, 0));
    M.winRecv.assign(C.N, 0);
    M.winDsum.assign(C.N, 0);

    if (!X.ns3mode)
    {
        for (int t = 0; t < C.T; ++t)
            StepEvent(t);
    }
    else
    {
        gUav.Create(C.N);
        gGw.Create(2);
        NodeContainer all(gUav, gGw);
        // PHY / channel
        Ptr<MultiModelSpectrumChannel> chan = CreateObject<MultiModelSpectrumChannel>();
        Ptr<LogDistancePropagationLossModel> ld = CreateObject<LogDistancePropagationLossModel>();
        ld->SetAttribute("Exponent", DoubleValue(C.plExp));
        ld->SetAttribute("ReferenceDistance", DoubleValue(C.d0));
        ld->SetAttribute("ReferenceLoss", DoubleValue(C.PLd0));
        Ptr<StepShadowLoss> sh = CreateObject<StepShadowLoss>();
        ld->SetNext(sh);
        UpdateShadow(X.net);
        chan->AddPropagationLossModel(ld);
        chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

        WifiHelper wifi;
        if (std::getenv("UAVRTS"))
            Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", UintegerValue(0));
        wifi.SetStandard(WIFI_STANDARD_80211n);
        wifi.SetRemoteStationManager("ns3::IdealWifiManager");
        WifiMacHelper mac;
        mac.SetType("ns3::AdhocWifiMac");
        const int chNum[3] = {1, 6, 11};
        std::vector<NetDeviceContainer> devs(gK);
        for (int k = 0; k < gK; ++k)
        {
            SpectrumWifiPhyHelper phy;
            phy.SetChannel(chan);
            phy.Set("ChannelSettings",
                    StringValue("{" + std::to_string(chNum[k]) + ", 20, BAND_2_4GHZ, 0}"));
            phy.Set("RxSensitivity", DoubleValue(C.sens));
            phy.Set("CcaSensitivity", DoubleValue(C.sens));
            phy.Set("TxPowerStart", DoubleValue(C.Pmax - 6));
            phy.Set("TxPowerEnd", DoubleValue(C.Pmax - 6));
            phy.Set("RxNoiseFigure", DoubleValue(6));
            devs[k] = wifi.Install(phy, mac, all);
        }
        MobilityHelper mob;
        mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mob.Install(all);
        for (int i = 0; i < C.N; ++i)
            gUav.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(s.px[i], s.py[i], C.z));
        for (int gw = 0; gw < 2; ++gw)
            gGw.Get(gw)->GetObject<MobilityModel>()->SetPosition(Vector(s.gx[gw], s.gy[gw], s.gz[gw]));
        InternetStackHelper st;
        Ipv4StaticRoutingHelper srh;
        st.SetRoutingHelper(srh);
        st.Install(all);
        gAddr.assign(C.N + 2, std::vector<Ipv4Address>(gK));
        for (int k = 0; k < gK; ++k)
        {
            Ipv4AddressHelper ah;
            ah.SetBase(("10." + std::to_string(k + 1) + ".0.0").c_str(), "255.255.0.0");
            Ipv4InterfaceContainer ifs = ah.Assign(devs[k]);
            for (uint32_t j = 0; j < all.GetN(); ++j)
                gAddr[j][k] = ifs.GetAddress(j);
        }
        NeighborCacheHelper nch;
        nch.PopulateNeighborCache();
        // sockets
        TypeId udp = UdpSocketFactory::GetTypeId();
        for (int gw = 0; gw < 2; ++gw)
        {
            Ptr<Socket> rs = Socket::CreateSocket(gGw.Get(gw), udp);
            rs->Bind(InetSocketAddress(Ipv4Address::GetAny(), kPort));
            rs->SetRecvCallback(MakeCallback(&RecvCb));
        }
        gTx.resize(C.N);
        gRateMbps.assign(C.N, 0);
        gDest.assign(C.N, -1);
        for (int i = 0; i < C.N; ++i)
        {
            gTx[i] = Socket::CreateSocket(gUav.Get(i), udp);
            gTx[i]->Bind();
            Simulator::Schedule(Seconds(0.001 + 0.0005 * i), &SendPacket, i);
        }
        Simulator::Schedule(Seconds(0.0), &StepEvent, 0);
        Simulator::Stop(Seconds(C.T * C.dt + C.Dmax / 1000 + 0.1));
        Simulator::Run();
        Simulator::Destroy();
    }

    if (std::getenv("UAVDBG") && X.ns3mode)
    {
        for (int i = 0; i < C.N; ++i)
        {
            double sn = 0, rc = 0;
            for (int t = 0; t < C.T; ++t)
            {
                sn += M.sent[t][i];
                rc += M.recv[t][i];
            }
            std::cerr << "uav " << i << " sent " << sn << " recv " << rc << " load " << X.net.load[i] << " hop "
                      << X.net.hop[i] << " nh " << X.net.nextHop[i] << " dst " << X.net.destGw[i] << " ch "
                      << X.s.ch[i] << " P " << X.s.P[i] << " pos " << X.s.px[i] << "," << X.s.py[i] << "\n";
        }
    }
    // ---- output: per step model metrics and measured metrics
    std::ofstream o(outFn);
    o << "step,m_delay,m_pdr,m_thr,m_fair,loadvar,handover";
    if (X.ns3mode)
        o << ",delay,pdr,thr,fair,sent,recv,pktdelay";
    o << "\n";
    for (int t = 0; t < C.T; ++t)
    {
        const auto& r = X.rows[t];
        for (size_t k = 0; k < r.size(); ++k)
            o << (k ? "," : "") << r[k];
        if (X.ns3mode)
        {
            int N = C.N, act = 0;
            double dsum = 0, psum = 0, thr = 0, s1 = 0, s2 = 0, sent = 0, recv = 0, pk = 0;
            for (int i = 0; i < N; ++i)
            {
                double sn = M.sent[t][i], rc = M.recv[t][i];
                sent += sn;
                recv += rc;
                pk += M.dsum[t][i];
                double th = rc * C.Tpacket / C.dt / 1e6; // Mbps
                thr += th;
                s1 += th;
                s2 += th * th;
                if (sn > 0)
                {
                    ++act;
                    psum += std::min(rc / sn, 1.0);
                    dsum += (rc > 0) ? M.dsum[t][i] / rc : C.Dmax;
                }
            }
            double fair = s1 * s1 / (N * s2 + 1e-9);
            o << "," << (act ? dsum / act : 0) << "," << (act ? psum / act : 0) << "," << thr << "," << fair
              << "," << sent << "," << recv << "," << (recv > 0 ? pk / recv : C.Dmax);
        }
        o << "\n";
    }
    return 0;
}
