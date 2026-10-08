import unittest

from experimentguard import diagnose
from experimentguard.sample_data import generate


def run(scenario, **kw):
    return diagnose(*generate(scenario, n_users=30_000, **kw))


def finding(result, prefix):
    return next(f for f in result["findings"] if f["id"].startswith(prefix) and f["severity"] != "pass")


class ScenarioTests(unittest.TestCase):
    def test_healthy_experiment_is_trustworthy(self):
        r = run("healthy")
        self.assertEqual(r["verdict"], "TRUSTWORTHY")
        self.assertEqual(r["recommended_action"], "proceed")
        self.assertFalse(r["needs_human_approval"])

    def test_healthy_experiment_stays_clean_across_seeds(self):
        for seed in range(1, 8):
            self.assertEqual(run("healthy", seed=seed)["verdict"], "TRUSTWORTHY", f"false positive at seed {seed}")

    def test_exposure_loss_is_localized_in_time_and_segment(self):
        r = run("exposure_loss")
        self.assertEqual(r["verdict"], "UNRELIABLE")
        f = finding(r, "exposure_gap")
        self.assertEqual(f["evidence"]["onset"], "2026-09-08")  # a Tuesday
        top = f["evidence"]["top_segments"][0]
        self.assertEqual((top["dimension"], top["segment"]), ("device+app_version", "ios / 3.2"))
        self.assertTrue(r["needs_human_approval"])
        self.assertIn("a.device = 'ios'", f["sql"])

    def test_exposure_loss_does_not_trigger_allocation_srm(self):
        r = run("exposure_loss")
        self.assertTrue(all(f["severity"] == "pass" for f in r["findings"] if f["id"] == "allocation"))

    def test_srm_dropout_finds_segment_and_onset(self):
        r = run("srm_dropout")
        f = finding(r, "allocation")
        self.assertEqual(f["evidence"]["onset"], "2026-09-09")
        self.assertEqual(f["evidence"]["top_segments"][0]["segment"], "paid")

    def test_duplicates_found_and_attributed_to_treatment(self):
        r = run("duplicates")
        f = finding(r, "duplicates")
        self.assertEqual(f["severity"], "critical")
        self.assertIn("'treatment'", f["summary"])
        self.assertEqual(f["evidence"]["onset"], "2026-09-08")

    def test_crossover_flags_identity_problems(self):
        r = run("crossover")
        ids = {f["id"] for f in r["findings"] if f["severity"] == "critical"}
        self.assertIn("identity_multi_assignment", ids)
        self.assertIn("identity_exposure_mismatch", ids)
        self.assertEqual(r["recommended_action"], "restart_experiment")


class InputTests(unittest.TestCase):
    def test_uneven_split_is_respected(self):
        a, e = generate("healthy", n_users=20_000)
        # keep only 10% of treatment users to simulate a deliberate 90/10 split
        keep = a[(a.variant == "control") | (a.index % 9 == 0)]
        r = diagnose(keep, e[e.user_id.isin(keep.user_id)], split="90/10")
        self.assertTrue(all(f["severity"] == "pass" for f in r["findings"] if f["id"] == "allocation"), r["findings"][0])

    def test_missing_columns_raise_clear_error(self):
        a, e = generate("healthy", n_users=1000)
        with self.assertRaisesRegex(ValueError, "assignments is missing"):
            diagnose(a.drop(columns="variant"), e)

    def test_works_without_optional_columns(self):
        a, e = generate("exposure_loss", n_users=20_000)
        r = diagnose(a[["user_id", "variant", "assigned_at"]], e[["user_id", "event_type", "timestamp"]])
        self.assertEqual(r["verdict"], "UNRELIABLE")  # still finds the gap, just can't localize to a segment
        self.assertEqual(finding(r, "exposure_gap")["evidence"]["onset"], "2026-09-08")


if __name__ == "__main__":
    unittest.main()
