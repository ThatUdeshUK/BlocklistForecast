# Blocklist-Forecast

The implementation of the paper [Blocklist-Forecast: Proactive Domain Blocklisting by Identifying Malicious Hosting Infrastructure](https://doi.ieeecomputersociety.org/).

## Abstract

Domain blocklists play an important role in blocking malicious domains reaching users. However, existing blocklists are reactive in nature and slow to react to attacks, by which time the damage is already caused. This is mainly due to the fact that existing blocklists and reputation systems rely on either website content or user interactions with the websites in order to ascertain if a website is malicious. In this work, we explore the possibility of predicting malicious domains proactively, given a seed list of malicious domains from such reactive blocklists. We observe that malicious domains often share the infrastructure utilized for previous attacks, reuse or rotate resources. Leveraging this observation, we selectively crawl passive DNS data to identify domains in the "neighborhood" of seed malicious domains extracted from reactive blocklists. Due to the increased utilization of cloud hosting, not all such domains in the neighborhood are malicious. Further vetting is required to identify unseen malicious domains. Along with the proximity, we identify that hosting and lexical features help distinguish malicious domains from benign ones. We model the infrastructure as a heterogeneous network graph and design a graph neural network to detect malicious domains. Our approach is blocklist-agnostic in that it can work with any blocklist and detect new malicious domains. We demonstrate our approach utilizing 7 month longitudinal data from three popular blocklists, PhishTank, OpenPhish, and VirusTotal. Our experimental results show that, our approach for VirusTotal feed detects 4.7 unseen malicious domains for every seed malicious domain at a very low FPR of 0.059. Further, we observe the concerning trend that 47% of predicted malicious domains that are later flagged in VirusTotal are identified only after more than 3 weeks to months since our model detects them.

<img width="1450" alt="overview" src="https://github.com/user-attachments/assets/324645ec-02c5-48ab-9232-b32401e11561">

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email (`mail@udesh.xyz`). If you encounter any problems when using the code, or want to report a bug, you can open an issue.

<!--
## Citation

Please cite our paper if you use your work:

```bibtex

```
->
