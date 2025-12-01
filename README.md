This project investigates the sociotechnical implications of using synthetic data to "fix" algorithmic fairness. While data augmentation is a popular technique to balance datasets and improve statistical metrics (like demographic parity), 
we argue that it often introduces a fidelity paradox: the model improves on paper, but relies on stereotypical or low-fidelity representations of marginalized groups.

Using Large Language Models (LLMs), we simulate a "Fairness-as-Augmentation" pipeline to demonstrate:

The Illusion of the Fix: How synthetic text improves standard fairness metrics.
The Sociotechnical Reality: A qualitative audit revealing that this "fairness" comes at the cost of reinforcing stereotypes (e.g., synthetic female surgeon bios emphasizing "care" over "competence").

We use data from https://huggingface.co/datasets/LabHC/bias_in_bios under the MIT license.


Our approach follows a three-phase audit protocol:

Phase 1: Quantitative Baseline (The Problem) We train a baseline classifier (Logistic Regression/TF-IDF) on a biased dataset (e.g., professional bios) and establish that it underperforms for minority groups.
Phase 2: The "Fix" (Augmentation) We use an LLM (e.g., GPT-4o/Llama 3) to generate synthetic examples for the underrepresented class. We retrain the model and demonstrate a statistical improvement in Demographic Parity and Equalized Odds.
Phase 3: The Paradox (Qualitative Audit) We perform a sociotechnical audit of the synthetic data, analyzing linguistic patterns to measure Fidelity and stereotypical harm, using frameworks from Buolamwini, Gebru, and Crawford.
