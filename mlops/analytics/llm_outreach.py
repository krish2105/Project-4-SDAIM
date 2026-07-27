def generate_llm_retention_outreach(customer_dict, churn_prob, top_shap_drivers):
    """
    Generates personalized customer retention outreach copy (Email & SMS).
    
    Args:
        customer_dict: Customer attributes (Age, Geography, Balance, etc.)
        churn_prob: Churn probability score
        top_shap_drivers: List of top SHAP risk factors
        
    Returns:
        dict containing Subject, Email_Body, and SMS_Copy.
    """
    geo = customer_dict.get('Geography', 'France')
    tenure = customer_dict.get('Tenure', 3)
    balance = customer_dict.get('Balance', 50000.0)
    num_prod = customer_dict.get('NumOfProducts', 1)
    
    subject = f"Special Relationship Offer for Your Bank Account"
    
    email_body = f"""Subject: {subject}

Dear Valued Customer,

Thank you for being a trusted banking client with us for over {tenure} years in {geo}. 

We recently reviewed your relationship profile and noticed that your current account balance of ${balance:,.2f} qualifies you for our exclusive Premier Banking Relationship Package.

To show our appreciation for your continued trust, we would like to offer you the following complimentary benefits:
  • 🌟 Promotional 4.25% APY Bonus Interest Rate on your savings balance.
  • 💳 100% Annual Maintenance & Credit Card Fee Waiver for the next 12 months.
  • 📞 Direct access to a Dedicated Personal Relationship Manager.

If you have any questions or would like to activate these benefits immediately, please reply to this email or contact your personal advisor at your local {geo} branch.

Warm regards,

Executive Relationship Team
Bank Customer Services
"""

    sms_copy = f"Bank Alert: Exclusive offer for your {tenure}-year relationship! Enjoy a 4.25% APY bonus & waived account fees. Contact your branch advisor today."
    
    return {
        'Subject': subject,
        'Email_Body': email_body,
        'SMS_Copy': sms_copy
    }
