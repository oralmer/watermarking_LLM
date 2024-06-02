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

    @classmethod
    def generate_from_params(
        cls,
        word_len: int,
        random_bits: int,
        num_conditions: int,
        condition_sparseness: int,
        noise_probability: float,
        failure_rate: float,
    ) -> (np.array, np.array):
        rng = default_rng()
        P = binary_field.Zeros((num_conditions, word_len), dtype=np.int8)
        for i in range(num_conditions):
            P[i, rng.choice(word_len, size=condition_sparseness, replace=False)] = 1
        null_basis = P.null_space()
        G = binary_field.Zeros((word_len, random_bits))
        for i in range(random_bits):
            rand_vec = binary_field.Random(null_basis.shape[0])
            G[:, i] = null_basis.T @ rand_vec
        assert np.all(P @ G == 0)
        z = binary_field.Random(word_len)
        return cls(
            P=P,
            G=G,
            z=z,
            noise_probability=noise_probability,
            failure_rate=failure_rate,
        )

    def generate_codeword(self):
        u = binary_field.Random(self.G.shape[1])
        e = binary_field(
            np.random.choice(
                [0, 1],
                size=(self.G.shape[0],),
                p=[1 - self.noise_probability, self.noise_probability],
            )
        )
        return (self.G @ u) + self.z + e

    def validate_codeword(self, x: GF):
        return (
            np.array((self.P @ x) + (self.P @ self.z)).sum()
            < self.failure_rate * self.P.shape[0]
        )


if __name__ == "__main__":
    for i in range(20):
        prc = PRC.generate_from_params(50, 10, 50, 5, 0, 0.1)
        word = prc.generate_codeword()
        rand = binary_field.Random(prc.G.shape[0])
        print(prc.validate_codeword(word), prc.validate_codeword(rand))
