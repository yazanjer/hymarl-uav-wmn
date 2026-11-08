function o = uwmn_getobs(env, useAug)
%UWMN_GETOBS  Observation selector (augmented graph obs vs plain obs).
if useAug, o = env.getObsAug(); else, o = env.getObs(); end
end
