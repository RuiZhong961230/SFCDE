# SFCDE
Competitive differential evolution with success-failure adaptation mechanism: performance benchmarking and application in eggplant disease diagnosis

## Abstract
Competitive Differential Evolution (CDE) often suffers from suboptimal performance, being hampered by static parameter settings and an inefficient exploration-exploitation balance. In this study, Success-Failure Competitive Differential Evolution (SFCDE), a novel variant of DE, is introduced to address these challenges through the incorporation of a Success-Failure Adaptation Mechanism (SFA). By dynamically adjusting the scaling factor (F) and crossover rate (Cr) based on both historical success patterns (to reinforce effective parameter configurations) and failure patterns (to evade unproductive search regions), SFCDE enhances convergence speed and robustness across diverse optimization landscapes. The core contributions encompass: (1) the success-failure mechanism adjusts F and Cr in real time, striking a balance between exploiting known good solutions and exploring new search spaces; (2) SFCDE is evaluated on CEC2017/2020/2022 benchmarks as well as seven engineering problems, outperforming nine state-of-the-art metaheuristics in terms of solution accuracy and convergence efficiency; (3) an SFCDE-Ensemble method is developed for eggplant disease diagnosis. By fine-tuning eight pre-trained deep models on a Kaggle dataset, selecting the top three performers based on performance metrics, and optimizing soft voting weights using SFCDE, the ensemble achieves an accuracy of 99.68%, precision of 99.69%, recall of 99.68%, and an F1-score of 99.68%—markedly surpassing single models. These results underscore SFCDE’s efficacy in overcoming static parameter limitations and its scalability to real-world problems such as agricultural disease diagnosis. The source code is publicly accessible at https://github.com/RuiZhong961230/SFCDE, facilitating reproducibility and further research endeavors.

## Citation
@article{Wang:26,  
  title={Competitive differential evolution with success-failure adaptation mechanism: performance benchmarking and application in eggplant disease diagnosis},  
  author={Zhongmin Wang and Daihong Li and Jun Yu and Mahmoud Abdel-Salam and Gang Hu and Essam H. Houssein and Nagwan Abdel Samee and Rui Zhong },  
  journal={International Journal of Machine Learning and Cybernetics},  
  pages={},  
  volume={17},  
  year={2026},  
  publisher={Springer},  
  doi = {https://doi.org/10.1007/s13042-025-02931-3 },  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. The eggplant disease diagnosis dataset can be downloaded at https://www.kaggle.com/datasets/kamalmoha/eggplant-disease-recognition-dataset

## Contact
If you have any questions, please don't hesitate to contact zhongminwang[at]ynau.edu.cn and zhongrui[at]iic.hokudai.ac.jp
