# g0.py
# Distribuição G^0 (intensidade) para imagens SAR
# Baseada diretamente na definição analítica do artigo
#
# Parâmetros:
#   alpha < 0   (rugosidade)
#   gamma > 0   (escala)
#   L >= 1      (número de looks)
#   z > 0       (intensidade)

import math
from scipy.special import gamma, gammaln


def g0_pdf(z, alpha, gamma_p, L):
    """
    Densidade de probabilidade da distribuição G^0 (intensidade):

        f(z) =
        [ L^L * Gamma(L - alpha) / (gamma^alpha * Gamma(-alpha) * Gamma(L)) ]
        * z^(L-1) * (gamma + L z)^(alpha - L)

    válida para z > 0
    """

    if alpha >= 0:
        raise ValueError("alpha deve ser < 0")
    if gamma_p <= 0:
        raise ValueError("gamma deve ser > 0")
    if L < 1:
        raise ValueError("L deve ser >= 1")

    if z <= 0:
        return 0.0

    C = (L**L * gamma(L - alpha)) / (
        (gamma_p**alpha) * gamma(-alpha) * gamma(L)
    )

    return C * (z ** (L - 1)) * ((gamma_p + L * z) ** (alpha - L))


def g0_logpdf(z, alpha, gamma_p, L):
    """
    Logaritmo da densidade G^0.
    Usado apenas para estabilidade numérica.
    """

    if alpha >= 0:
        raise ValueError("alpha deve ser < 0")
    if gamma_p <= 0:
        raise ValueError("gamma deve ser > 0")
    if L < 1:
        raise ValueError("L deve ser >= 1")

    if z <= 0:
        return float("-inf")

    logC = (
        L * math.log(L)
        + gammaln(L - alpha)
        - alpha * math.log(gamma_p)
        - gammaln(-alpha)
        - gammaln(L)
    )

    logV = (L - 1) * math.log(z) + (alpha - L) * math.log(gamma_p + L * z)

    return logC + logV
