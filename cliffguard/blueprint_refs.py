"""Mapping from every CLIFFGUARD module path to its blueprint section."""

BLUEPRINT_REFS: dict[str, str] = {
    # Component __init__ packages
    "cliffguard.vestibule": "§5.6, §5.7",
    "cliffguard.probe": "§5.1, §5.2, §5.3",
    "cliffguard.bprobe": "§5.10, §5.11",
    "cliffguard.tripwire": "§5.4, §5.5",
    "cliffguard.conductor": "§6",
    "cliffguard.lookout": "§5.8, §5.9",
    "cliffguard.ladder": "§10",
    "cliffguard.attest": "§5.12",
    # Primitive modules
    "cliffguard.vestibule.lz": "§5.6",
    "cliffguard.vestibule.ps": "§5.7",
    "cliffguard.probe.rm": "§5.1",
    "cliffguard.probe.mt": "§5.2",
    "cliffguard.probe.hd": "§5.3",
    "cliffguard.bprobe.logit": "§5.10",
    "cliffguard.bprobe.consistency": "§5.11",
    "cliffguard.tripwire.h": "§5.4",
    "cliffguard.tripwire.r": "§5.5",
    "cliffguard.conductor.bandit": "§6",
    "cliffguard.lookout.ct": "§5.8",
    "cliffguard.lookout.jg": "§5.9",
    "cliffguard.ladder.tier": "§10",
    "cliffguard.ladder.router": "§10",
    "cliffguard.attest.wh": "§5.12",
}
