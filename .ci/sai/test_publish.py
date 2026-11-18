import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish as site


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.repo = {"id": 1, "full_name": site.REPO}
        fork = {"id": 2, "full_name": "contributor/LibDDLA"}
        self.run = dict(id=17, run_attempt=1, status="in_progress", event="pull_request",
                        path=site.WORKFLOW, repository=self.repo, head_repository=fork,
                        head_branch="topic", head_sha=self.sha, display_title=f"SAI CUDA / 3 / {self.sha}")
        self.pull = dict(state="open", head=dict(sha=self.sha, repo=fork),
                         base=dict(ref="develop", repo=self.repo))
        self.routes = {
            "actions/runs/17": self.run, "pulls/3": self.pull,
            "actions/runs/17/approvals": [{"state": "approved", "user": {"login": "owner"},
                                           "environments": [{"name": site.REVIEW_ENV}]}],
            "actions/runs/17/attempts/1/jobs": {"jobs": [{"status": "in_progress",
                                                          "labels": [site.RESULT_LABEL]}]},
        }
        mock = patch.object(site, "api", side_effect=self.routes.__getitem__)
        self.api = mock.start()
        self.addCleanup(mock.stop)

    def test_pr_needs_current_head_and_its_own_approval(self):
        self.assertEqual(site.admission(self.run, {"owner"}), ("develop", self.sha, "3"))
        with self.assertRaises(ValueError):
            site.admission(self.run, {"other"})
        self.pull["head"]["sha"] = "b" * 40
        with self.assertRaises(ValueError):
            site.admission(self.run, {"owner"})
        self.run.update(id=18, head_sha="b" * 40, display_title=f"SAI CUDA / 3 / {'b' * 40}")
        self.routes["actions/runs/18/approvals"] = []
        with self.assertRaises(ValueError):
            site.admission(self.run, {"owner"})

    def test_push_and_manual_sources(self):
        self.run.update(event="push", head_repository=self.repo, head_branch="main",
                        display_title=f"SAI CUDA / 0 / {self.sha}")
        self.routes[f"compare/{self.sha}...main"] = {"status": "ahead"}
        self.assertEqual(site.admission(self.run, set()), ("main", self.sha, "0"))
        for path, bad in [(f"compare/{self.sha}...main", {"status": "diverged"})]:
            with self.subTest(path=path), patch.dict(self.routes, {path: bad}), self.assertRaises(ValueError):
                site.admission(self.run, set())
        self.run.update(event="workflow_dispatch", display_title=f"SAI CUDA / 0 / {'b' * 40}")
        self.assertEqual(site.admission(self.run, {"owner"}), ("main", "b" * 40, "0"))
        self.pull["base"]["ref"] = "topic"
        self.run.update(event="pull_request", head_repository=self.pull["head"]["repo"],
                        display_title=f"SAI CUDA / 3 / {self.sha}")
        with self.assertRaises(ValueError):
            site.admission(self.run, {"owner"})

    def test_request_tokens_are_scoped_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("work", "results", "requests"):
                (root / folder).mkdir()
            token = "ghs_" + "a" * 36
            request = root / "requests/gh-17-1.token"
            request.write_text(token)
            env = {"SAI_WORK": str(root / "work"), "SAI_RESULTS": str(root / "results"),
                   "SAI_REQUESTS": str(root / "requests"), "SAI_REVIEWERS": "owner"}
            with patch.dict(os.environ, env), patch.object(site, "publish") as publish:
                publish.side_effect = lambda *args: self.assertEqual(os.environ["GH_TOKEN"], token)
                site.main()
                publish.assert_called_once_with(self.run, {"owner"})
                self.assertNotIn("GH_TOKEN", os.environ)
                publish.reset_mock()
                request.unlink()
                request.symlink_to(root / "secret")
                (root / "secret").write_text(token)
                site.main()
                publish.assert_not_called()

    def test_receipt_and_cancellation(self):
        for exit_status in (0, 1, "cancel"):
            with self.subTest(exit_status=exit_status), tempfile.TemporaryDirectory() as tmp:
                env = dict(SAI_RESULTS=tmp, SAI_WORK=tmp + "/work", SAI_RELAY="/relay",
                           SAI_SSH_CONFIG="/config", SAI_AUTHORIZE="")
                relay = Mock()
                if exit_status == "cancel":
                    relay.wait.side_effect = [subprocess.TimeoutExpired("relay", 20), 1]
                    relay.poll.return_value = None
                    self.pull["state"] = "open"
                    def change_state(*args, **kwargs):
                        self.pull["state"] = "closed"
                        return relay
                else:
                    relay.wait.return_value = exit_status
                    relay.poll.return_value = exit_status
                    change_state = lambda *args, **kwargs: relay
                with patch.dict(os.environ, env), patch.object(site.subprocess, "Popen") as start:
                    start.return_value.__enter__.side_effect = change_state
                    site.publish(self.run, {"owner"})
                    site.publish(self.run, {"owner"})
                    self.assertEqual(start.call_count, 1)
                    args = start.call_args.args[0]
                    self.assertIn(self.sha, args)
                    receipt = json.loads((Path(tmp) / "gh-17-1/completion.json").read_text())
                    self.assertEqual(receipt, {"run_id": "gh-17-1", "source_sha": self.sha,
                                               "exit_code": 0 if exit_status == 0 else 1})
                    if exit_status == "cancel":
                        relay.send_signal.assert_called_once_with(site.signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
