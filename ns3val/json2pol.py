"""Convert an exported MATLAB actor (JSON) into the plain-text format read by uavval.cc."""
import json, sys
def conv(src, dst):
    p = json.load(open(src))
    L = p["layers"]
    if isinstance(L, dict): L = [L]
    out = [p["method"], p["assign"], p["chanMode"], p["powerMode"], p["dObs"], p["actDim"], p["contDim"],
           p["chanN"], p["powN"], p["K"], int(p["aug"]), int(p["useLyap"]), 1, p["kL"], p["Ttarget"],
           p["dPmax"], p["dS"], len(L)]
    lines = [" ".join(str(v) for v in out)]
    for l in L:
        W = l["W"]; b = l["b"]
        if not isinstance(W[0], list): W = [W]
        if not isinstance(b, list): b = [b]
        lines.append(f"{len(W)} {len(W[0])}")
        lines.append(" ".join(repr(float(x)) for row in W for x in row))
        lines.append(" ".join(repr(float(x)) for x in b))
    open(dst, "w").write("\n".join(lines) + "\n")
if __name__ == "__main__":
    conv(sys.argv[1], sys.argv[2])
