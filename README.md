# OOTM-Option-Pricing-with-Power-Laws-and-News

__*note: for work more representative of my current abilities please see my Discord-Stonks repository. This was just an exploration of Talebs ideas.*__

In Nassim Taleb's Antifragile he mentions that CEOs featured in Forbe's (mostly "fragilistas") can be overconfident and blind to tails risks. He even claims that there is alpha in betting against these people. I want to test this statement (along with several others) and see whether public statements and media effects the fair price of OOTM options.

In the fair_options_pricing.ipynb file, I use math published in Nassim Taleb's Statistical Consequences of Fat Tails book to assign reasonable prices to OOTM options. The pricing framework focuses on power law distributions and how accurately estimating their parameters helps lower uncertainty around their value. I built a scraper for Forbes articles and will use sentiment analysis to determine whether any factors indicate systemic risk and change the "fair" price of long-term OOTM options.

In the HawkesProcesses.ipynb file, I use Taleb's statistical framework as a foundation for valuing near-term OOTM Technology stock options with expirations following earnings reports. I build a bivariate Hawkes process with exponential marks and a power law kernel. I then build a framework to test the statistical signifance of hawkes process fits. Based on earnings relases, I will predict the parameters of jump diffusion hawkes processes model and use the defined stochastic process to value OOTM options via simulation.

For designing the Hawkes process framework, I relied most heavily on the following sources:
* https://arxiv.org/abs/1502.04592 (great literature overview)
* https://arxiv.org/abs/1302.1405 (uses HPs w/ power laws to measure critical reflexivity in markets)
* https://people.math.ethz.ch/~embrecht/ftp/Hawkes_PE_TL_LL.pdf (Looks into multivariate point processes with vector-valued marks) 
* https://arxiv.org/abs/1405.6047 (FX markets w/ news)
* https://link.springer.com/content/pdf/10.1007/BF02480272.pdf (derivation of MLE jacobian)
