Credit Scoring Bussiness Understanding
1. Basel II and Model Interpretability

The Basel II Capital Accord mandates rigorous and transparent risk management practices for financial institutions. This emphasis on auditable risk measurement directly influences our model choice. The model must be interpretable—meaning Bati Bank can clearly and simply explain why a customer was approved or denied a loan. This is critical for:

Regulatory Compliance: Meeting audit requirements by proving the model is fair and non-discriminatory.
Transparency: Providing customers with a justifiable reason for the credit decision.

Example: A simple model like Logistic Regression allows us to say, "You were denied because your average transaction frequency (a key feature) falls below the risk threshold," which is compliant.
2. Proxy Variable Necessity and Business Risks

Necessity: We lack a direct "default" label because the data is from an e-commerce partner, not a banking institution. Creating a proxy variable using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering is necessary to artificially create a high-risk/low-risk target for our classification model.


Potential Business Risks: The proxy is an assumption based on behavior, not a verified outcome. This introduces the risk of two costly errors:

False Positives (Approving Bad Loans): We approve a customer labeled "low-risk" by the proxy who actually defaults. This leads to direct financial loss for Bati Bank.

False Negatives (Denying Good Customers): We deny a customer labeled "high-risk" by the proxy who would have paid back the loan. This leads to lost business, lost revenue, and damage to customer relations.


3. Model Trade-Offs in a Regulated Context

Model Type | Key Trade-Off | Why it Matters for Bati Bank
"Simple (e.g., Logistic Regression)" |  High Interpretability for loan justification.  Lower Performance (potentially higher risk exposure).|  Preferred for its transparency and regulatory compliance with Basel II.
__________________________________________________________________________________________________________________
"Complex (e.g., Gradient Boosting)"|" Higher Performance (better accuracy/lower default rate). Low Interpretability (""Black Box"")."|"Achieves better results but is difficult to audit and explain, posing a regulatory risk."