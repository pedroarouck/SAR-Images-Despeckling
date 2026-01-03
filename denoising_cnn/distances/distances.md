## Referencial teórico

O conteúdo apresentado nesta página tem como base a Seção II — The $G^0$ Distribution for Speckled Data do artigo:

Nascimento, A. D. C.; Cintra, R. J.; Frery, A. C.
*Hypothesis Testing in Speckled Data With Stochastic Distances*.
IEEE Transactions on Geoscience and Remote Sensing, vol. 48, n. 1, p. 373–385, 2010.

## Ruído speckle e modelo multiplicativo

O ruído speckle, característico de sistemas de imageamento coerente, não possui natureza Gaussiana nem aditiva. Esse comportamento decorre do próprio mecanismo de formação da imagem, no qual o sinal registrado resulta da interferência coerente de múltiplas contribuições elementares dentro de cada célula de resolução.

Nesse contexto, o modelo mais adequado para a análise estatística de dados afetados por speckle é o modelo multiplicativo, que emerge diretamente da física do processo de imageamento e descreve com precisão o sinal de retorno em imagens SAR.

Segundo esse modelo, cada elemento da imagem é descrito como a realização de uma variável aleatória $Z$, denominada retorno, definida como o produto de duas variáveis aleatórias independentes:

$$
Z = X \cdot Y
$$

A variável aleatória $X$ representa o retroespalhamento do terreno, enquanto a variável aleatória $Y$ representa o ruído speckle associado ao processo de aquisição coerente.

Embora sistemas coerentes possam fornecer informações complexas por pixel, neste trabalho considera-se exclusivamente o formato de intensidade do sinal de retorno, por ser o mais utilizado em aplicações práticas.

## Modelagem estatística das variáveis aleatórias $X$, $Y$ e $Z$

O retroespalhamento do terreno concentra toda a informação relevante da área mapeada, estando diretamente relacionado às propriedades físicas do alvo. Uma distribuição adequada para modelar estatisticamente essa componente é a distribuição Gamma inversa. Assim, assume-se que a variável aleatória $X$ segue a distribuição

$$
X \sim \Gamma^{-1}(\alpha, \gamma),
$$

cuja função densidade de probabilidade é dada por

$$
f_X(x;\alpha,\gamma) =
\frac{\gamma^{-\alpha}}{\Gamma(-\alpha)}
x^{\alpha-1}
\exp\left(-\frac{\gamma}{x}\right),
\quad -\alpha > 0,; \gamma > 0,; x > 0.
$$

Essa parametrização corresponde a um caso particular da distribuição Gaussiana Inversa Generalizada.

O ruído speckle $Y$ apresenta distribuição exponencial com média unitária em imagens SAR de intensidade com um único look. Quando é aplicado um procedimento de multi-look sobre $L$ observações independentes, o speckle em intensidade passa a ser descrito por uma distribuição Gamma, isto é,

$$
Y \sim \Gamma(L, L),
$$

com função densidade de probabilidade dada por

$$
f_Y(y;L) =
\frac{L^L}{\Gamma(L)}
y^{L-1}
\exp(-Ly),
\quad y > 0,; L \ge 1.
$$

Neste trabalho, o número de looks $L$ é assumido conhecido e constante em toda a imagem.

Considerando as distribuições definidas acima para $X$ e $Y$, bem como a independência estatística entre essas variáveis aleatórias, a distribuição associada à variável de retorno

$$
Z = X Y
$$

pode ser derivada. A variável aleatória $Z$ segue então a distribuição $G^0$, cuja função densidade de probabilidade é dada por

$$
f_Z(z;\alpha,\gamma,L) =
\frac{L^L \Gamma(L-\alpha)}
{\gamma^{\alpha}\Gamma(-\alpha)\Gamma(L)}
;
\frac{z^{L-1}}
{(\gamma + Lz)^{L-\alpha}},
\quad -\alpha > 0,; \gamma > 0,; z > 0,; L \ge 1.
$$

## Propriedades e abrangência do modelo $G^0$

A variável aleatória de retorno $Z$ que resulta do modelo multiplicativo é denotada por

$$
Z \sim G^0(\alpha, \gamma, L).
$$

A distribuição $G^0$ pode ser utilizada como um modelo universal para dados afetados por ruído speckle.

O momento de ordem $r$ da variável aleatória $Z$ é dado por

$$
\mathbb{E}[Z^r] =
\left(\frac{\gamma}{L}\right)^r
\frac{\Gamma(-\alpha - r)}{\Gamma(-\alpha)}
\frac{\Gamma(L + r)}{\Gamma(L)},
$$

desde que $-r > \alpha$. Caso contrário, o momento é infinito. Essa propriedade evidencia a influência direta do parâmetro de rugosidade $\alpha$ na existência dos momentos da distribuição, refletindo o grau de heterogeneidade do terreno.

Diversos métodos para a estimação dos parâmetros $\alpha$ e $\gamma$ são descritos na literatura. Neste trabalho, devido às suas propriedades assintóticas ótimas, adota-se a estimação por máxima verossimilhança, cuja solução requer procedimentos numéricos.

## Estimação dos parâmetros do modelo $G^0$

Considere uma amostra aleatória de tamanho $n$, denotada por
$z = (z_1, z_2, \dots, z_n)$, em que cada observação $z_i$ segue a distribuição $G^0(\alpha,\gamma,L)$.

A função de verossimilhança associada ao modelo $G^0(\alpha,\gamma,L)$ é dada por

$$
\mathcal{L}(\alpha,\gamma; z) =
\left(
\frac{L^L \Gamma(L-\alpha)}
{\gamma^{\alpha}\Gamma(-\alpha)\Gamma(L)}
\right)^n
\prod_{i=1}^{n}
z_i^{,L-1}
(\gamma + L z_i)^{\alpha - L}.
$$

Os estimadores de máxima verossimilhança de $\alpha$ e $\gamma$, denotados por $\hat{\alpha}$ e $\hat{\gamma}$, são obtidos como solução do seguinte sistema de equações não lineares:

$$
\begin{cases}
\psi_0(L - \hat{\alpha}) - \psi_0(-\hat{\alpha}) - \log(\hat{\gamma})

* \dfrac{1}{n}\sum_{i=1}^{n}\log(\hat{\gamma} + L z_i) = 0, [10pt]
  -\dfrac{\hat{\alpha}}{\hat{\gamma}}
* \dfrac{\hat{\alpha}-L}{n}
  \sum_{i=1}^{n}(\hat{\gamma} + L z_i)^{-1} = 0,
  \end{cases}
  $$

em que $\psi_0(\cdot)$ denota a função digama.

Esse sistema não admite, em geral, solução em forma fechada, sendo necessário o uso de métodos numéricos de otimização para a obtenção dos estimadores. No artigo de referência, é adotado o método BFGS, amplamente reconhecido por sua eficiência e precisão.



