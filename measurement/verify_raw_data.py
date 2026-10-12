"""Independent check, straight from the raw StatsBomb JSON.
Deliberately imports nothing from src/ -- if this agrees with the pipeline,
the pipeline is right; if it disagrees, the pipeline is wrong."""
import json, glob, statistics as st
from collections import Counter

FILES = sorted(glob.glob("open-data/data/events/*.json"))[:200]
PITCH = 120.0
OPTA   = {"Interception","Foul Committed","Block","Clearance"}
DRAFT  = OPTA | {"Pressure"}

lit_tackle = duel_tackle = 0
shot_x, gk_x, p1_x, p2_x = [], [], [], []
ppda = {"opta":[], "draft":[]}
zero_windows = {"opta":0, "draft":0}; tot_windows = 0

def defensive(e, types):
    t = e["type"]["name"]
    if t in types: return True
    return t == "Duel" and e.get("duel",{}).get("type",{}).get("name") == "Tackle"

for fp in FILES:
    with open(fp) as f: ev = json.load(f)
    teams = list({e["team"]["id"] for e in ev if "team" in e})
    if len(teams) != 2: continue
    for e in ev:
        t = e["type"]["name"]
        if t == "Tackle": lit_tackle += 1
        if t == "Duel" and e.get("duel",{}).get("type",{}).get("name")=="Tackle": duel_tackle += 1
        loc = e.get("location")
        if loc:
            if t == "Shot": shot_x.append(loc[0])
            if e.get("position",{}).get("name")=="Goalkeeper": gk_x.append(loc[0])
            (p1_x if e.get("period")==1 else p2_x).append(loc[0])
    for press, opp in ((teams[0],teams[1]),(teams[1],teams[0])):
        for key, types in (("opta",OPTA),("draft",DRAFT)):
            passes = sum(1 for e in ev if e["team"]["id"]==opp and e["type"]["name"]=="Pass"
                         and e.get("location") and e["location"][0] < PITCH-40)
            acts   = sum(1 for e in ev if e["team"]["id"]==press and defensive(e,types)
                         and e.get("location") and e["location"][0] > 40)
            if acts: ppda[key].append(passes/acts)
    # 5-minute windows, pressing team = teams[0]
    for start in range(0, 90, 5):
        tot_windows += 1
        for key, types in (("opta",OPTA),("draft",DRAFT)):
            a = sum(1 for e in ev if e["team"]["id"]==teams[0] and defensive(e,types)
                    and e.get("minute") is not None and start <= e["minute"] < start+5
                    and e.get("location") and e["location"][0] > 40)
            if a == 0: zero_windows[key] += 1

def med(x): return round(st.median(x),2) if x else None
print(f"matches parsed        : {len(FILES)}")
print()
print("GATE 1  tackles")
print(f"  literal 'Tackle' events   : {lit_tackle}")
print(f"  tackles inside Duel       : {duel_tackle}   <- must be > 0")
print()
print("GATE 2  coordinates already normalized (no flip needed)")
print(f"  shot x  median : {med(shot_x)}   (should be HIGH, ~100+)")
print(f"  GK   x  median : {med(gk_x)}     (should be LOW,  ~5)")
print(f"  period1 x med  : {med(p1_x)}")
print(f"  period2 x med  : {med(p2_x)}     (should MATCH period 1)")
print()
print("GATE 3  match-level PPDA  (published range 8-15)")
print(f"  Opta standard (no Pressure) : median {med(ppda['opta'])}   n={len(ppda['opta'])}")
print(f"  draft spec    (w/ Pressure) : median {med(ppda['draft'])}   n={len(ppda['draft'])}")
print()
print("GATE 4  5-min windows with ZERO defensive actions")
for k in ("opta","draft"):
    print(f"  {k:<6}: {zero_windows[k]}/{tot_windows} = {100*zero_windows[k]/tot_windows:.1f}%")
