"""
tests/test_ablation_configs.py
================================
Verifies that all ablation configuration definitions are valid, complete,
and internally consistent.

No LLM calls. No file I/O. Pure schema tests.
"""

import pytest
from evaluation.ablation_study.configs import (
    ALL_CONFIGS,
    UNIQUE_CONFIGS,
    CONFIG_BY_ID,
    ARCH_CONFIGS,
    RULE_CONFIGS,
    THRESHOLD_CONFIGS,
    get_config,
    list_config_ids,
)

REQUIRED_KEYS = {
    "id", "axis", "label", "description",
    "disabled_rules", "disabled_flags",
    "suitable_threshold", "conditional_threshold",
    "is_control",
}

VALID_AXES = {"architecture", "rule", "threshold"}
VALID_FLAGS = {"BORDERLINE", "CONCENTRATION", "CONTRADICTION", "ESCALATION"}
VALID_RULES = {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}


class TestConfigSchema:
    def test_all_configs_have_required_keys(self):
        for cfg in ALL_CONFIGS:
            missing = REQUIRED_KEYS - cfg.keys()
            assert not missing, f"Config {cfg.get('id')} missing keys: {missing}"

    def test_axis_values_are_valid(self):
        for cfg in ALL_CONFIGS:
            assert cfg["axis"] in VALID_AXES, \
                f"Config {cfg['id']} has invalid axis: {cfg['axis']}"

    def test_disabled_flags_are_valid(self):
        for cfg in ALL_CONFIGS:
            bad = set(cfg["disabled_flags"]) - VALID_FLAGS
            assert not bad, \
                f"Config {cfg['id']} has invalid disabled_flags: {bad}"

    def test_disabled_rules_are_valid(self):
        for cfg in ALL_CONFIGS:
            bad = set(cfg["disabled_rules"]) - VALID_RULES
            assert not bad, \
                f"Config {cfg['id']} has invalid disabled_rules: {bad}"

    def test_thresholds_are_integers(self):
        for cfg in ALL_CONFIGS:
            assert isinstance(cfg["suitable_threshold"], int)
            assert isinstance(cfg["conditional_threshold"], int)

    def test_suitable_threshold_above_conditional(self):
        for cfg in ALL_CONFIGS:
            assert cfg["suitable_threshold"] > cfg["conditional_threshold"], \
                f"Config {cfg['id']}: suitable_threshold must be > conditional_threshold"

    def test_is_control_is_bool(self):
        for cfg in ALL_CONFIGS:
            assert isinstance(cfg["is_control"], bool), \
                f"Config {cfg['id']}: is_control must be bool"

    def test_id_is_string(self):
        for cfg in ALL_CONFIGS:
            assert isinstance(cfg["id"], str) and cfg["id"]


class TestConfigCounts:
    def test_arch_configs_count(self):
        assert len(ARCH_CONFIGS) == 5, \
            f"Expected 5 architecture configs, got {len(ARCH_CONFIGS)}"

    def test_rule_configs_count(self):
        assert len(RULE_CONFIGS) == 8, \
            f"Expected 8 rule configs, got {len(RULE_CONFIGS)}"

    def test_threshold_configs_count(self):
        assert len(THRESHOLD_CONFIGS) == 3, \
            f"Expected 3 threshold configs, got {len(THRESHOLD_CONFIGS)}"

    def test_all_configs_total(self):
        assert len(ALL_CONFIGS) == 16, \
            f"Expected 16 total configs, got {len(ALL_CONFIGS)}"

    def test_unique_configs_excludes_duplicate_controls(self):
        # R0 and T0 are control duplicates of A0 — excluded from UNIQUE_CONFIGS
        ids = [c["id"] for c in UNIQUE_CONFIGS]
        assert "R0" not in ids, "R0 should be excluded from UNIQUE_CONFIGS (duplicate control)"
        assert "T0" not in ids, "T0 should be excluded from UNIQUE_CONFIGS (duplicate control)"
        assert "A0" in ids, "A0 (control) must be in UNIQUE_CONFIGS"

    def test_unique_configs_count(self):
        # 16 total − 2 duplicate controls (R0, T0) = 14
        assert len(UNIQUE_CONFIGS) == 14, \
            f"Expected 14 unique configs, got {len(UNIQUE_CONFIGS)}"


class TestConfigIds:
    def test_all_ids_unique_across_all_configs(self):
        ids = [c["id"] for c in ALL_CONFIGS]
        assert len(ids) == len(set(ids)), \
            f"Duplicate config IDs found: {[i for i in ids if ids.count(i) > 1]}"

    def test_config_by_id_contains_all(self):
        for cfg in ALL_CONFIGS:
            assert cfg["id"] in CONFIG_BY_ID

    def test_get_config_returns_correct(self):
        cfg = get_config("A0")
        assert cfg["id"] == "A0"
        assert cfg["is_control"] is True

    def test_get_config_raises_on_unknown_id(self):
        with pytest.raises(KeyError):
            get_config("NONEXISTENT_XYZ")

    def test_list_config_ids_matches_unique_configs(self):
        ids = list_config_ids()
        expected = [c["id"] for c in UNIQUE_CONFIGS]
        assert ids == expected


class TestControlConfigs:
    def test_exactly_three_controls_in_all_configs(self):
        controls = [c for c in ALL_CONFIGS if c["is_control"]]
        assert len(controls) == 3, \
            f"Expected 3 controls (A0, R0, T0), got {[c['id'] for c in controls]}"

    def test_controls_are_a0_r0_t0(self):
        controls = {c["id"] for c in ALL_CONFIGS if c["is_control"]}
        assert controls == {"A0", "R0", "T0"}

    def test_control_configs_have_no_disabled_rules(self):
        for cfg in ALL_CONFIGS:
            if cfg["is_control"]:
                assert cfg["disabled_rules"] == [], \
                    f"Control {cfg['id']} should have no disabled_rules"

    def test_control_configs_have_no_disabled_flags(self):
        for cfg in ALL_CONFIGS:
            if cfg["is_control"]:
                assert cfg["disabled_flags"] == [], \
                    f"Control {cfg['id']} should have no disabled_flags"

    def test_control_configs_have_default_thresholds(self):
        for cfg in ALL_CONFIGS:
            if cfg["is_control"]:
                assert cfg["suitable_threshold"]    == 70
                assert cfg["conditional_threshold"] == 40


class TestAblationSemantics:
    def test_a1_disables_all_flags(self):
        cfg = get_config("A1")
        assert set(cfg["disabled_flags"]) == VALID_FLAGS

    def test_a2_disables_only_borderline(self):
        cfg = get_config("A2")
        assert cfg["disabled_flags"] == ["BORDERLINE"]

    def test_a3_disables_only_concentration(self):
        cfg = get_config("A3")
        assert cfg["disabled_flags"] == ["CONCENTRATION"]

    def test_a4_disables_only_contradiction(self):
        cfg = get_config("A4")
        assert cfg["disabled_flags"] == ["CONTRADICTION"]

    def test_each_rule_config_disables_exactly_one_rule(self):
        single_rule_configs = [c for c in RULE_CONFIGS if not c["is_control"]]
        for cfg in single_rule_configs:
            assert len(cfg["disabled_rules"]) == 1, \
                f"Config {cfg['id']} should disable exactly 1 rule"

    def test_t1_is_stricter_than_default(self):
        t1 = get_config("T1")
        assert t1["suitable_threshold"]    > 70
        assert t1["conditional_threshold"] > 40

    def test_t2_is_more_lenient_than_default(self):
        t2 = get_config("T2")
        assert t2["suitable_threshold"]    < 70
        assert t2["conditional_threshold"] < 40
