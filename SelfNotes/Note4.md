# CS229 Machine Learning Note 4
# Generative learning algorithms
**Discriminative learning algorithms**: Algorithms that try to learn $p(y|x)$ directly of try to learn mappings directly from the space of inputs.
**Generative learning algorithms**: Algorithms that try to learn $p(x|y)$ and $p(y)$(**class priors 先验**) and then use Bayes' rule to compute $p(y|x)$. Thus deriving posterior distribution by 
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
where the denominator is not that important since:
$$
\argmax_{y} p(y|x) = \argmax_y p(x|y) p(y)
$$

## 4.1 Gaussian discriminant analysis (GDA)
In this model, assume $p(x|y)$ is distributed according to a multivariate normal distribution.
### 4.1.1 The multivariate normal distribution
**Multivariate distribution** in $d$-dimensions, also as **multivariate Gaussian distribution**: 
  Parameterized by 
  - **mean vector** $\mu \in \mathbb{R}^d$  
  - **covariance matrix** $\Sigma \in \mathbb{R}^{d \times d}$, where $\Sigma \geq 0 $ is **symmetric** and **positive semi-definite**($x^TMx \geq 0$)
  - written as $\mathcal{N}(\mu, \Sigma)$

$$
p(x;\mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right), \quad \text{where} |\Sigma| = \det \Sigma
$$
- mean:
  $$
  E[X] = \int_x x p(x;\mu, \Sigma)dx = \mu
  $$
- **covariance** of vec valued random variable $Z$: $\text{Cov}(Z) = E[(Z-E[Z])(Z-E[Z])^T]$, which is equivalent to $E[ZZ^T]-(E[Z])(E[Z])^T$
- Intuitively, a Gaussian with $\mu = 0, \Sigma = I$ is called **standard normal distribution**
  
### 4.1.2 The Gaussian discriminant analysis model
Given a classification problem where input features $x$ are continuous-valued random variables, we may use GDA to model $p(x|y)$.
$$
y \sim \text{Bernoulli}(\phi)
\\x|y = 0 \sim \mathcal{N}(\mu_0, \Sigma)\\
x|y = 1 \sim \mathcal{N}(\mu_1, \Sigma)
$$
Thus:
$$
p(y) = \phi^y (1-\phi)^{1-y}
$$
$$
p(x|y = 0) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x-\mu_0)^T \Sigma^{-1} (x-\mu_0)\right)
$$
$$
p(x|y = 1) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1)\right)
$$
Parameters of model are $\phi, \Sigma, \mu_0, \mu_1$, thus derive the log-likelihood function:
$$
\begin{align*}
\ell(\phi, \Sigma, \mu_0, \mu_1) &= \log \prod_{i=1}^{n} p(x^{(i)},y^{(i)}; \phi, \Sigma, \mu_0, \mu_1)\\
&= \log \prod_{i=1}^{n} p(x^{(i)}|y^{(i)};\Sigma, \mu_0, \mu_1)p(y^{(i)};\phi)
\end{align*}
$$
The MLE of parameters is:
$$
\begin{align*}
\phi &= \frac{1}{n} \sum_{i=1}^{n} 1\{y^{(i)} = 1\}\\
\mu_0 &= \frac{\sum_{i=1}^{n} 1\{y^{(i)} = 0\}x^{(i)}}{\sum_{i=1}^{n}1\{y^{(i)} = 0\}}\\
\mu_1 &= \frac{\sum_{i=1}^{n} 1\{y^{(i)} = 1\}x^{(i)}}{\sum_{i=1}^{n}1\{y^{(i)} = 1\}}\\
\Sigma &= \frac{1}{n} \sum_{i=1}^{n} (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
\end{align*}
$$