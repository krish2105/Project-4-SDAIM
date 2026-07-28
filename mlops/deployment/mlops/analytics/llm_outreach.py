def generate_llm_retention_outreach(customer_dict, churn_prob, top_shap_drivers):
    """
    Generates personalized e-commerce customer retention outreach copy (Email & SMS).
    """
    cat = customer_dict.get('PreferedOrderCat', 'Laptop & Accessory')
    payment = customer_dict.get('PreferredPaymentMode', 'Debit Card')
    tenure = customer_dict.get('Tenure', 12)
    cashback = customer_dict.get('CashBackAmount', 150.0)
    
    subject = f"Exclusive VIP Discount & $50 CashBack on Your Favorite {cat} Products!"
    
    email_body = f"""Subject: {subject}

Dear Valued Shopper,

Thank you for shopping with us over the past {tenure} months! 

We noticed you frequently browse our {cat} section using {payment}. To show our appreciation for your loyalty, we have added an exclusive $50 VIP Reward Credit to your account!

🌟 Your Special Loyalty Perks:
  • 🎁 $50 CashBack Bonus automatically applied to your next checkout.
  • 🚚 Free Express Priority Delivery on all {cat} orders this month.
  • ⚡ 20% Instant Discount on all electronics & lifestyle essentials.

Claim your reward today at checkout before it expires!

Warm regards,

Customer Success & Loyalty Team
E-Commerce Shopping Platform
"""

    sms_copy = f"VIP Loyalty Perk: Enjoy $50 bonus CashBack + Free Priority Delivery on your favorite {cat} products! Shop now before your credit expires."
    
    return {
        'Subject': subject,
        'Email_Body': email_body,
        'SMS_Copy': sms_copy
    }
