from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng
from galois import GF

binary_field = GF(2)


@dataclass(frozen=True)
class PRC:
    P: GF
    G: GF
    z: GF
    noise_probability: float
    failure_rate: float

    @property
    def word_length(self) -> int:
        return self.G.shape[0]

    @property
    def num_conditions(self) -> int:
        return self.P.shape[0]

    @classmethod
    def generate_from_params(
        cls,
        word_length: int,
        random_bits: int,
        num_conditions: int,
        condition_sparseness: int,
        noise_probability: float,
        failure_rate: float,
    ) -> "PRC":
        assert word_length > num_conditions
        assert word_length > random_bits
        rng = default_rng()
        P = binary_field.Zeros((num_conditions, word_length), dtype=np.int8)
        while np.linalg.matrix_rank(P) != num_conditions:
            for i in range(num_conditions):
                P[
                    i, rng.choice(word_length, size=condition_sparseness, replace=False)
                ] = 1
        null_basis = P.null_space()
        G = binary_field.Zeros((word_length, random_bits))
        for i in range(random_bits):
            rand_vec = binary_field.Random(null_basis.shape[0])
            G[:, i] = null_basis.T @ rand_vec
        assert np.all(P @ G == 0)
        z = binary_field.Random(word_length)
        return cls(
            P=P,
            G=G,
            z=z,
            noise_probability=noise_probability,
            failure_rate=failure_rate,
        )

    def generate_codeword(self) -> GF:
        u = binary_field.Random(self.G.shape[1])
        e = binary_field(
            np.random.choice(
                [0, 1],
                size=(self.word_length,),
                p=[1 - self.noise_probability, self.noise_probability],
            )
        )
        return (self.G @ u) + self.z + e

    def validate_codeword(self, x: GF) -> bool:
        return self.score_codeword(x) < self.failure_rate * self.num_conditions

    def score_codeword(self, x: GF) -> float:
        return np.array((self.P @ x) + (self.P @ self.z)).sum()


def get_success_ratio(prc: PRC, num_checks=100) -> float:
    successes = 0
    fails = 0
    for i in range(num_checks):
        word = prc.generate_codeword()
        rand = binary_field.Random(prc.word_length)
        successes += int(prc.validate_codeword(word))
        fails += int(prc.validate_codeword(rand))
    print("fails: ", fails, fails / num_checks)
    print("successes: ", successes, successes / num_checks)
    return successes / (fails + successes + 1)


def get_scores(prc: PRC, num_checks=100) -> (float, float):
    successes = 0
    fails = 0
    for i in range(num_checks):
        word = prc.generate_codeword()
        rand = binary_field.Random(prc.word_length)
        successes += prc.score_codeword(word) / prc.num_conditions
        fails += prc.score_codeword(rand) / prc.num_conditions
    print("fails: ", fails / num_checks)
    print("successes: ", successes / num_checks)
    return successes, fails


if __name__ == "__main__":
    import nevergrad as ng

    def target(
        random_bits: int,
        num_conditions: int,
        condition_sparseness: int,
        failure_rate: float,
        num_regens: int = 3,
        num_checks: int = 1000,
    ):
        try:
            # We want to minimize so it's negative.
            return (
                sum(
                    -get_success_ratio(
                        PRC.generate_from_params(
                            16 * 4,
                            random_bits,
                            num_conditions,
                            condition_sparseness,
                            0.3,
                            failure_rate,
                        ),
                        num_checks=num_checks // num_regens,
                    )
                    for _ in range(num_regens)
                )
                / num_regens
            )
        except Exception as e:
            print(e)
            return 0

    # for _ in range(1):
    #     target(30, 40, 2, 0.4)
    get_scores(PRC.generate_from_params(16 * 6, 30, 70, 3, 0.2, 0.4), 3000)
    exit(1)
    # Discrete, ordered
    random_bits = ng.p.TransitionChoice(range(5, 70))
    num_conditions = ng.p.TransitionChoice(range(16, 70))
    condition_sparseness = ng.p.TransitionChoice(range(3, 17, 2))
    failure_rate = ng.p.Scalar(lower=0, upper=0.5)
    params = ng.p.Instrumentation(
        random_bits, num_conditions, condition_sparseness, failure_rate
    )
    optimizer = ng.optimizers.DiscreteOnePlusOne(
        parametrization=params, budget=500, num_workers=1
    )

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss = target(*x.value[0])
        print(loss, x.value[0])
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    print(recommendation.value, recommendation.loss)
