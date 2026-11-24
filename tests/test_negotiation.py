import unittest

import numpy as np
import torch

from nsng_energy.agents import Beliefs
from nsng_energy.negotiation import GraphInputs, Negotiator, Settings, graph_inputs


def _inputs(length, seed):
    scores = np.random.default_rng(seed).random((length, 7)).astype(np.float32)
    return graph_inputs(Beliefs(scores, 0.5 + np.abs(scores - 0.5)), None, None, torch.device("cpu"))


class NegotiationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Negotiator(Settings()).eval()

    @torch.inference_mode()
    def test_consensus_follows_equations_18_and_19(self):
        consensus, confidence, details, _ = self.model(_inputs(40, 1), trace=True)
        final = details["states"][0, torch.arange(40), details["rounds"][0].long()]
        score, agent_confidence, trust = final.unbind(-1)
        weights = agent_confidence * trust
        expected = (weights * score).sum(-1) / (weights.sum(-1) + 1e-8)
        agreement = 1 - (score - expected[:, None]).abs() / 2
        expected_confidence = (weights * agreement).sum(-1) / (weights.sum(-1) + 1e-8)
        torch.testing.assert_close(consensus[0], expected)
        torch.testing.assert_close(confidence[0], expected_confidence)

    @torch.inference_mode()
    def test_outgoing_weights_are_row_stochastic(self):
        _, _, details, _ = self.model(_inputs(30, 2), trace=True)
        rounds = details["rounds"][0].long()
        for t in range(30):
            rows = details["outgoing"][0, t, :rounds[t]].sum(-1)
            torch.testing.assert_close(rows, torch.ones_like(rows))

    @torch.inference_mode()
    def test_trust_is_carried_across_time(self):
        data = _inputs(50, 3)
        whole = self.model(data)
        first = self.model(data.take(slice(0, 20)))
        second = self.model(data.take(slice(20, 50)), state=first[3])
        torch.testing.assert_close(whole[0][:, 20:], second[0])
        self.assertFalse(torch.allclose(whole[2]["trust"][0, -1], torch.full((7,), 0.8)))

    @torch.inference_mode()
    def test_padding_does_not_change_results(self):
        short, long = _inputs(25, 4), _inputs(40, 5)
        alone = self.model(short)[0]
        batched = self.model(GraphInputs.stack([short, long]))[0]
        torch.testing.assert_close(batched[0, :25], alone[0])

    @torch.inference_mode()
    def test_trust_ablation_keeps_initial_trust(self):
        _, _, details, _ = self.model(_inputs(20, 6), Settings(trust=False))
        torch.testing.assert_close(details["trust"], torch.full_like(details["trust"], 0.8))


if __name__ == "__main__":
    unittest.main()
