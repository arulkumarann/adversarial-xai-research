"""Reference slides."""
from __future__ import annotations

from .content_static import S
from .lib import *  # noqa: F401,F403

MAIN = [
    "Szegedy et al. (2014). Intriguing properties of neural networks. ICLR. arXiv:1312.6199.",
    "Goodfellow, Shlens, Szegedy (2015). Explaining and harnessing adversarial examples. ICLR. arXiv:1412.6572.",
    "Papernot et al. (2016). The limitations of deep learning in adversarial settings. IEEE EuroS&P.",
    "Moosavi-Dezfooli, Fawzi, Frossard (2016). DeepFool: a simple and accurate method to fool deep neural networks. CVPR.",
    "Kurakin, Goodfellow, Bengio (2017). Adversarial examples in the physical world. ICLR Workshop. arXiv:1607.02533.",
    "Carlini, Wagner (2017). Towards evaluating the robustness of neural networks. IEEE S&P.",
    "Madry et al. (2018). Towards deep learning models resistant to adversarial attacks. ICLR. arXiv:1706.06083.",
    "Sabour, Cao, Faghri, Fleet (2016). Adversarial manipulation of deep representations. ICLR.",
    "Yu, Gao, Xu (2021). LAFEAT: piercing through adversarial defenses with latent features. CVPR.",
    "Zhou et al. (2016). Learning deep features for discriminative localization. CVPR.",
    "Selvaraju et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV.",
    "Ribeiro, Singh, Guestrin (2016). \"Why should I trust you?\": explaining the predictions of any classifier. KDD.",
    "Lundberg, Lee (2017). A unified approach to interpreting model predictions. NeurIPS.",
    "Simonyan, Vedaldi, Zisserman (2014). Deep inside convolutional networks: visualising image classification models and saliency maps. ICLR Workshop.",
    "Sundararajan, Taly, Yan (2017). Axiomatic attribution for deep networks. ICML.",
    "Huang, Liu, van der Maaten, Weinberger (2017). Densely connected convolutional networks. CVPR.",
    "Deng et al. (2009). ImageNet: a large-scale hierarchical image database. CVPR.",
    "Wang, Bovik, Sheikh, Simoncelli (2004). Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Processing 13(4).",
]
EXPL = [
    "Adebayo et al. (2018). Sanity checks for saliency maps. NeurIPS.",
    "Ghorbani, Abid, Zou (2019). Interpretation of neural networks is fragile. AAAI.",
    "Dombrowski et al. (2019). Explanations can be manipulated and geometry is to blame. NeurIPS.",
    "Subramanya, Pillai, Pirsiavash (2019). Fooling network interpretation in image classification. ICCV.",
    "Heo, Joo, Moon (2019). Fooling neural network interpretations via adversarial model manipulation. NeurIPS.",
    "Slack et al. (2020). Fooling LIME and SHAP: adversarial attacks on post hoc explanation methods. AIES.",
    "Zhang et al. (2020). Interpretable deep learning under fire. USENIX Security.",
    "Viering et al. (2019). How to manipulate CNNs to make them lie: the GradCAM case. BMVC Workshop.",
    "Noppel, Peter, Wressnegger (2023). Disguising attacks with explanation-aware backdoors. IEEE S&P.",
    "Baniecki, Biecek (2024). Adversarial attacks and defenses in explainable artificial intelligence: a survey. Information Fusion 107:102303.",
    "Kindermans et al. (2019). The (un)reliability of saliency methods. In Explainable AI, LNCS 11700.",
    "Dimanov et al. (2020). You shouldn't trust me: learning models which conceal unfairness from multiple explanation methods. ECAI.",
    "Anders et al. (2020). Fairwashing explanations with off-manifold detergent. ICML.",
    "Le Merrer, Trédan (2020). Remote explainability faces the bouncer problem. Nature Machine Intelligence.",
    "Lakkaraju, Bastani (2020). \"How do I fool you?\": manipulating user trust via misleading black box explanations. AIES.",
    "Zhang et al. (2021). Data poisoning attacks against outcome interpretations of predictive models. KDD.",
    "Baniecki, Kretowicz, Biecek (2022). Fooling partial dependence via data poisoning. ECML PKDD.",
    "Laberge et al. (2023). Fooling SHAP with stealthily biased sampling. ICLR.",
]
DEF = [
    "Rieger, Hansen (2020). A simple defense against adversarial attacks on heatmap explanations. ICML Workshop on Human Interpretability.",
    "Chen et al. (2019). Robust attribution regularization. NeurIPS.",
    "Boopathy et al. (2020). Proper network interpretability helps adversarial robustness in classification. ICML.",
    "Wang et al. (2020). Smoothed geometry for robust attribution. NeurIPS.",
    "Dombrowski et al. (2022). Towards robust explanations for deep neural networks. Pattern Recognition.",
    "Joo et al. (2023). Towards more robust interpretation via local gradient alignment. AAAI.",
    "Liu et al. (2022). Certifiably robust interpretation via Rényi differential privacy. Artificial Intelligence.",
    "Wicker et al. (2023). Robust explanation constraints for neural networks. ICLR.",
    "Ghalebikesabi et al. (2021). On locality of local explanation models. NeurIPS.",
    "Vreš, Robnik-Šikonja (2022). Preventing deception with explanation methods using focused sampling. Data Mining and Knowledge Discovery.",
    "Blesch et al. (2023). Unfooling SHAP and SAGE: knockoff imputation for Shapley values. xAI.",
    "Carmichael, Scheirer (2023). Unfooling perturbation-based post hoc explainers. AAAI.",
    "Hooker et al. (2019). A benchmark for interpretability methods in deep neural networks. NeurIPS.",
    "Hedström et al. (2023). Quantus: an explainable AI toolkit for responsible evaluation of neural network explanations. JMLR 24(34).",
    "Agarwal et al. (2022). OpenXAI: towards a transparent evaluation of model explanations. NeurIPS Datasets and Benchmarks.",
    "Arras, Osman, Samek (2022). CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion 81.",
    "Huang et al. (2023). SAFARI: versatile and efficient evaluations for robustness of interpretability. ICCV.",
    "Tabassi et al. (2019). A taxonomy and terminology of adversarial machine learning. NISTIR 8269 draft (withdrawn 2023; superseded by NIST AI 100-2e2023).",
]
DATA = "Image sample: EliSchwartz/imagenet-sample-images, GitHub (one ImageNet image per class)."


def refs(prs):
    for part, (title, items) in enumerate([("References: attacks, explanation methods, data and metrics", MAIN),
                                           ("References: attacks on explanations", EXPL),
                                           ("References: defences, evaluation and standards", DEF + [DATA])], 1):
        s = S(prs, f"07  |  References ({part}/3)", title, size=26)
        n = len(items)
        half = (n + 1) // 2
        for c, chunk in enumerate((items[:half], items[half:])):
            tb(s, MX + c * 6.05, 1.75, 5.85, 5.1,
               [{"text": f"{i + 1 + c * half}.  {t}", "after": 5} for i, t in enumerate(chunk)],
               size=12, color=G1, line=1.02)
