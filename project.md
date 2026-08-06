Here's the full content of the PDF you uploaded, as extracted:

---

**MSc Project Plan**
**CSC-44120**

**Project Overview and Description**

- Student Name: Muhammad Haseeb
- Student Username: y6w19
- Student Number: 25038315
- Degree Title: MSc AI and Data Science
- Supervisor Name: Dr Taimur ul Haq
- Project Title: *(not filled in)*

**Project Description:**
The increasing use of digital payment systems has made credit card fraud a significant concern for financial institutions and customers. As fraudulent activities become more sophisticated, conventional rule-based detection methods often struggle to identify emerging fraud patterns while maintaining a low rate of false alarms. This project aims to design and develop an AI-powered credit card fraud detection and alert system that improves the accuracy and efficiency of identifying suspicious transactions.

The proposed system will use machine learning techniques to analyse transaction data and distinguish legitimate transactions from potentially fraudulent ones. The project will include data preprocessing, feature engineering, model development, performance evaluation, and comparison of multiple machine learning algorithms to identify the most effective approach. Because fraud datasets are typically highly imbalanced, suitable techniques will be applied to improve the model's ability to detect fraudulent transactions without compromising overall performance.

To enhance the practical value of the solution, the system will simulate real-time fraud alerts and present transaction insights through an interactive dashboard. Model performance will be assessed using appropriate evaluation metrics, including precision, recall, F1-score, and ROC-AUC. The final outcome will be a reliable, scalable, and intelligent fraud detection framework that demonstrates how artificial intelligence can strengthen financial security and support timely decision-making in credit card fraud prevention.

**Aims and Objectives**

*Aim:*
The aim of this project is to design, develop, and evaluate an AI-powered credit card fraud detection and alert system that accurately identifies fraudulent transactions, minimizes false positives, and supports timely fraud prevention through intelligent decision-making.

*Objectives:*
1. To investigate current credit card fraud detection methods and identify their limitations.
2. To collect and preprocess a credit card transaction dataset, including handling missing values, feature scaling, and class imbalance.
3. To develop and train multiple AI and machine learning models for detecting fraudulent credit card transactions.
4. To compare the performance of the developed models using evaluation metrics such as precision, recall, F1-score, ROC-AUC, and accuracy.
5. To design a real-time fraud alert mechanism that notifies users or financial institutions when suspicious transactions are detected.
6. To develop an interactive dashboard for visualizing transaction data, fraud predictions, and model performance.
7. To evaluate the effectiveness, reliability, and scalability of the proposed system and provide recommendations for future improvements.

**Key Literature Overview**

Recent research has shown that artificial intelligence and machine learning have significantly improved the detection of credit card fraud compared with traditional rule-based systems. Conventional fraud detection methods rely on predefined rules and manually created patterns, making them less effective against rapidly evolving and previously unseen fraudulent activities. As a result, researchers have increasingly focused on data-driven approaches that can automatically learn transaction patterns and adapt to new fraud behaviours.

Studies have demonstrated that machine learning algorithms such as Logistic Regression, Decision Trees, Random Forests, Support Vector Machines, and Extreme Gradient Boosting (XGBoost) can effectively classify legitimate and fraudulent transactions. Ensemble learning methods, particularly Random Forest and XGBoost, are frequently reported to achieve high detection accuracy while reducing false positive rates. More recently, deep learning techniques, including Artificial Neural Networks (ANNs), Autoencoders, and Long Short-Term Memory (LSTM) networks, have been explored for their ability to capture complex relationships within large-scale transaction data.

A recurring challenge identified in the literature is the highly imbalanced nature of credit card transaction datasets, where fraudulent transactions represent only a small proportion of the overall data. Researchers have addressed this issue using techniques such as the Synthetic Minority Oversampling Technique (SMOTE), undersampling, cost-sensitive learning, and anomaly detection methods to improve model performance on minority fraud cases.

Previous studies have primarily focused on maximising classification accuracy; however, recent work highlights the importance of evaluation metrics such as precision, recall, F1-score, and ROC-AUC, as these provide a more reliable assessment of fraud detection performance on imbalanced datasets. In addition, there is growing interest in explainable artificial intelligence (XAI), enabling financial institutions to understand and justify model predictions, thereby increasing trust and supporting regulatory compliance.

Building on these findings, this project will compare multiple AI-based classification techniques, apply suitable methods to address class imbalance, and integrate a real-time fraud alert mechanism with an interactive dashboard. The proposed approach aims to provide an accurate, practical, and scalable solution for enhancing credit card fraud detection while reducing false alarms and supporting timely decision-making.

**Project Processes and Methods**

*Methodology:*
This project will follow a machine learning and data-driven methodology to develop an AI-powered credit card fraud detection and alert system. The methodology will consist of several stages, including data collection, preprocessing, exploratory data analysis, model development, evaluation, and system implementation.

Initially, a suitable credit card transaction dataset will be collected and analyzed to understand transaction patterns and identify important characteristics associated with fraudulent activities. The data will then undergo preprocessing steps, including data cleaning, feature transformation, normalization, and handling of class imbalance. Best practices for imbalanced classification problems will be applied, such as using techniques like Synthetic Minority Oversampling Technique (SMOTE), appropriate data splitting, and avoiding data leakage during model training.

During the model development phase, different machine learning and artificial intelligence algorithms will be implemented and compared. These may include Logistic Regression, Random Forest, Support Vector Machine, XGBoost, and deep learning approaches such as Artificial Neural Networks. Hyperparameter optimization and cross-validation techniques will be applied to improve model performance and ensure reliable results.

The developed models will be evaluated using suitable performance metrics, including precision, recall, F1-score, ROC-AUC, and confusion matrix analysis. Since fraud detection focuses on identifying rare fraudulent cases, emphasis will be placed on improving recall while maintaining acceptable precision levels to reduce false alerts.

Finally, the best-performing model will be integrated into a prototype fraud alert system with a dashboard for monitoring transactions and displaying fraud predictions. The project will follow best practices in machine learning development, including reproducible experiments, proper model validation, ethical use of financial data, and clear documentation of methods and results.

*Data Collection Methods:*
This project will primarily use an existing publicly available credit card transaction dataset rather than collecting new data directly from individuals. The dataset will contain anonymized transaction records, including relevant features required for developing and evaluating AI-based fraud detection models. Using an existing dataset ensures compliance with privacy and ethical considerations, as no personally identifiable or sensitive customer information will be collected.

The selected dataset will be analyzed and prepared through data preprocessing techniques, including data cleaning, feature analysis, transformation, and handling class imbalance. Exploratory data analysis will be performed to understand transaction patterns and identify factors that contribute to fraudulent activities.

No questionnaires, interviews, card sorting exercises, or direct user participation methods will be required for this project. Instead, simulated transaction scenarios may be used during the testing phase to evaluate how the developed system responds to legitimate and suspicious transactions and how effectively the alert mechanism performs.

The use of publicly available anonymized data combined with simulated testing provides an appropriate and ethical approach for developing and validating an AI-powered credit card fraud detection system.

**Skills Required and Skills to be Developed**

During my MSc course, I have developed a range of technical and analytical skills that will support the successful completion of this project. These include programming skills, data analysis, database management, software development, and knowledge of artificial intelligence and machine learning concepts. I have gained experience with programming languages and tools used for data processing and model development, which will be applied to build and evaluate AI-based fraud detection models. In addition, my understanding of research methods, academic writing, and critical analysis will help in conducting the literature review, analyzing results, and presenting project findings effectively.

For this project, I will apply skills related to machine learning model development, data preprocessing, feature engineering, performance evaluation, and data visualization. Knowledge of statistical analysis and algorithm selection will be important for comparing different approaches and identifying the most suitable model for credit card fraud detection.

To successfully complete the project, I will further develop advanced skills in artificial intelligence and machine learning, particularly in handling highly imbalanced datasets and improving fraud detection performance. I will enhance my knowledge of deep learning techniques, explainable AI methods, model optimization, and real-time machine learning deployment. Additionally, I aim to improve my skills in developing interactive dashboards and integrating AI models into practical applications.

Developing these skills will enable me to create a reliable and efficient fraud detection system while strengthening my overall expertise in artificial intelligence, data science, and secure software development.

**Time and Resource Planning**

*Standard Departmental Hardware used?* YES
The project will primarily use standard departmental computing facilities, including university-provided computers and available software resources. The hardware requirements for this project are not expected to exceed standard departmental capabilities.

The project will involve developing and evaluating machine learning models using publicly available datasets. The required software tools, including Python, Jupyter Notebook, machine learning libraries (such as Scikit-learn, TensorFlow/PyTorch, and XGBoost), and data analysis tools, will be installed and used within the available computing environment.

If additional computational resources are required for training complex deep learning models, cloud-based platforms such as Google Colab or similar services may be considered to provide access to additional processing power.

*Project stages:*
1. Literature Review and Research Planning – Reviewing existing research on AI-based fraud detection techniques and defining the project methodology.
2. Data Collection and Preparation – Selecting a suitable dataset, performing data cleaning, exploratory analysis, and preparing data for model training.
3. Model Development and Testing – Implementing machine learning algorithms, optimizing models, and comparing their performance.
4. System Development – Developing the fraud alert mechanism and dashboard for displaying predictions and results.
5. Evaluation and Analysis – Assessing model performance using appropriate evaluation metrics and analyzing findings.
6. Documentation and Final Submission – Writing the dissertation report, preparing project documentation, and presenting the outcomes.

No specialized hardware or physical materials are required for this project. Optional cloud computing resources may be used if additional processing capability is needed for advanced AI model training.

*Software already available in department used?* YES
The main development environment will include Python-based tools for data analysis, machine learning model development, and evaluation.

Software and libraries listed:
- Python
- Jupyter Notebook / Google Colab
- Scikit-learn
- TensorFlow/PyTorch
- Pandas and NumPy
- Matplotlib and Seaborn
- XGBoost/LightGBM
- Streamlit or similar dashboard frameworks

Most of these tools are open-source and freely available, so no additional software licenses are expected to be required.

*Programming required?* YES

- Programming Language: Python
- IDEs: Jupyter Notebook, Visual Studio Code, Google Colab (optional)
- Libraries/Frameworks: Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch, XGBoost/LightGBM, Matplotlib/Seaborn, Imbalanced-learn (imblearn), Streamlit

The programming work will focus on building, testing, and evaluating AI-based fraud detection models while following best practices for machine learning development, including reproducible experiments, proper validation, and performance analysis.

**Table of Risks**

| ID | Description | Probability | Prevention | Remedy |
|---|---|---|---|---|
| R1 | Difficulty obtaining a suitable dataset or limitations in available transaction data | Low | Use reliable publicly available anonymised datasets and review dataset quality before starting development | Select an alternative suitable dataset or adjust the research scope according to available data |
| R2 | Highly imbalanced dataset causing poor fraud detection performance | High | Apply best practices such as SMOTE, undersampling, feature engineering, and use appropriate evaluation metrics | Test alternative balancing techniques and optimise model parameters to improve fraud detection results |
| R3 | Machine learning models producing inaccurate predictions or high false positive rates | Medium | Compare multiple AI algorithms, perform cross-validation, and apply hyperparameter optimisation | Select the best-performing model and refine feature selection or model configuration |
| R4 | Limited computational resources for training advanced AI models | Medium | Use efficient algorithms, optimise code, and utilise available university or cloud-based computing resources | Reduce model complexity or use alternative lightweight models such as Random Forest or XGBoost |
| R5 | Software library compatibility issues during development | Low | Use stable software versions, maintain documentation of dependencies, and regularly test the development environment | Use alternative libraries, reinstall compatible versions, or switch to backup environments such as Google Colab |
| R6 | Challenges in implementing real-time fraud alerts and dashboard functionality | Medium | Develop the system incrementally and test each component separately before integration | Implement a simulated alert system or simplify dashboard functionality while maintaining core project objectives |
| R7 | Time management issues affecting project completion | Medium | Create a detailed project schedule, set milestones, and regularly review progress | Prioritise essential tasks and adjust the project scope if necessary |
| R8 | Lack of understanding of advanced AI techniques such as deep learning or explainable AI | Medium | Conduct additional research, follow academic resources, and complete practical experiments | Use simpler machine learning approaches or seek guidance from the project supervisor |
| R9 | Data privacy and ethical concerns related to financial information | Low | Use only anonymized publicly available datasets and follow ethical research guidelines | Remove sensitive information and ensure all analysis complies with university research policies |

**Gantt Chart** *(present in the PDF as an image — milestones include: Project Planning & Initial Research; MILESTONE: Project Planning Completed; Literature Review & Requirement Analysis; Project Requirements and Methodology Design; MILESTONE: Requirements and Methodology Framework; Data Preprocessing and Exploratory Data Analysis; MILESTONE: Dataset Cleaned and Documented; Feature Engineering and Model Building; MILESTONE: Initial Models Trained; Model Testing, Tuning and Performance Evaluation; MILESTONE: Model Evaluation Complete; Fraud Alert System and Dashboard Development; MILESTONE: Alert and Dashboard System Working; Final Documentation and Project Review; MILESTONE: Final Submission Complete)*

**References (Harvard Style)**

- Awoyemi, J.O., Adetunmbi, A.O. and Oluwadare, S.A. (2017) 'Credit card fraud detection using machine learning techniques: A comparative analysis', *2017 International Conference on Computing Networking and Informatics (ICCNI)*, pp. 1–9.
- Bhattacharyya, S., Jha, S., Tharakunnel, K. and Westland, J.C. (2011) 'Data mining for credit card fraud: A comparative study', *Decision Support Systems*, 50(3), pp. 602–613.
- Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357.
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. and Bontempi, G. (2018) 'Credit card fraud detection: A realistic modelling and a novel learning strategy', *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), pp. 3784–3797.
- Dal Pozzolo, A., Caelen, O., Le Borgne, Y.A., Waterschoot, S. and Bontempi, G. (2015) 'Learned lessons in credit card fraud detection from a practitioner perspective', *Expert Systems with Applications*, 41(10), pp. 4915–4928.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016) *Deep Learning*. Cambridge, MA: MIT Press.
- Han, J., Kamber, M. and Pei, J. (2012) *Data Mining: Concepts and Techniques*. 3rd edn. Waltham, MA: Morgan Kaufmann.
- Kuhn, M. and Johnson, K. (2013) *Applied Predictive Modeling*. New York: Springer.
- Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.
- Ngai, E.W.T., Hu, Y., Wong, Y.H., Chen, Y. and Sun, X. (2011) 'The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature', *Decision Support Systems*, 50(3), pp. 559–569.
- Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', *PLOS ONE*, 10(3), pp. 1–21.
- Varmedja, D., Karanovic, M., Sladojevic, S., Arsenovic, M. and Anderla, A. (2019) 'Credit card fraud detection—Machine learning methods', *2019 18th International Symposium INFOTEH-JAHORINA (INFOTEH)*, pp. 1–5.
- Zhang, C., Wang, X. and Wang, X. (2020) 'A survey of machine learning techniques for credit card fraud detection', *Journal of Information Security and Applications*, 54, pp. 1–10.

**Submission Date:** *(not filled in)*

*Footer note: "PLEASE NOTE THAT SHOULD YOUR PROJECT UNDERGO ANY MAJOR CHANGES FOLLOWING THE SUBMISSION OF THIS PLAN YOU ARE EXPECTED TO SUBMIT AN UPDATED PLAN WHICH ACCURATELY REFLECTS YOUR PROJECT."*