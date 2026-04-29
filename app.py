diff --git a/app.py b/app.py
index a822d481d517e9816fc31984d8bb16ba56543abf..16eff77d62d9de19f301e0cd6d16a68cfb3c4480 100644
--- a/app.py
+++ b/app.py
@@ -819,72 +819,87 @@ with st.sidebar:
     st.markdown(f"## {T['title']}")
     st.caption("Smart fraud detection prototype")
 
     selected_lang = st.radio(
         "🌍 Language / Тіл / Язык",
         LANG_OPTIONS,
         index=LANG_OPTIONS.index(st.session_state.lang),
         horizontal=True,
         key="lang_selector"
     )
     if selected_lang != st.session_state.lang:
         st.session_state.lang = selected_lang
         st.rerun()
 
     st.divider()
 
     mode = st.selectbox(
         T["mode"],
         [T["sms"], T["call"], T["file"]]
     )
 
     threshold = st.slider(T["threshold"], 0.1, 0.9, 0.5, 0.05)
 
     st.divider()
     st.markdown(f"### 🧪 {T['demo']}")
-    demo = st.selectbox(
-        T["demo"],
-        ["Fraud SMS", "Fraud Call", "Fake Delivery", "Fake Prize", "Relative Scam", "Safe Message"]
-    )
+    demo = st.selectbox(
+        T["demo"],
+        [
+            "Fraud SMS",
+            "Fraud Call",
+            "Fake Delivery",
+            "Fake Prize",
+            "Relative Scam",
+            "Fake Job Offer",
+            "Marketplace Prepayment Scam",
+            "Fake Utility Debt",
+            "Investment Scam",
+            "Safe Message",
+        ]
+    )
 
     st.divider()
     st.markdown("### 🛡️ Project stack")
     st.markdown("• Logistic Regression")
     st.markdown("• Feature Engineering")
     st.markdown("• Domain Analysis")
     st.markdown("• Explainable AI")
     st.markdown("• Rule-based Risk Boost")
     st.markdown("• Real-life Scam Scenarios")
 
-demo_texts = {
-    "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
-    "Fraud Call": "Здравствуйте, я сотрудник службы безопасности банка. По вашему счету подозрительная операция. Назовите код из SMS, чтобы мы отменили перевод.",
-    "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
-    "Fake Delivery": "Ваша посылка задержана. Срочно оплатите таможенную пошлину по ссылке http://delivery-pay-online.xyz",
-    "Fake Prize": "Поздравляем! Вы выиграли приз. Для получения подарка введите номер карты и CVV.",
-    "Relative Scam": "Ваш родственник попал в аварию. Срочно переведите деньги, никому не говорите."
-}
+demo_texts = {
+    "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
+    "Fraud Call": "Здравствуйте, я сотрудник службы безопасности банка. По вашему счету подозрительная операция. Назовите код из SMS, чтобы мы отменили перевод.",
+    "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
+    "Fake Delivery": "Ваша посылка задержана. Срочно оплатите таможенную пошлину по ссылке http://delivery-pay-online.xyz",
+    "Fake Prize": "Поздравляем! Вы выиграли приз. Для получения подарка введите номер карты и CVV.",
+    "Relative Scam": "Ваш родственник попал в аварию. Срочно переведите деньги, никому не говорите.",
+    "Fake Job Offer": "Поздравляем, вы приняты на удаленную работу. Для оформления выплат отправьте фото удостоверения, номер карты и OTP-код из SMS.",
+    "Marketplace Prepayment Scam": "Здравствуйте, я покупатель с маркетплейса. Подтвердите получение оплаты: перейдите по ссылке https://safe-deal-confirm.top и введите данные карты.",
+    "Fake Utility Debt": "Уведомление ЖКХ: у вас долг за коммунальные услуги. Во избежание отключения света оплатите сегодня по ссылке http://pay-service-24.site.",
+    "Investment Scam": "Гарантированный доход 30% в неделю! Переведите деньги на инвестиционный счет и сообщите код подтверждения для активации."
+}
 
 # =========================
 # HERO
 # =========================
 logo_block = f'<img src="{LOGO_HTML}" class="site-logo">' if LOGO_HTML else '<div style="font-size:58px">🔐</div>'
 
 st.markdown(f"""
 <div class="hero">
     <div class="hero-grid">
         <div>
             <div class="logo-row">
                 {logo_block}
                 <div>
                     <div class="hero-kicker">⚡ AI-powered safety scanner</div>
                     <div class="logo-title"><span>AI</span> FRAUD<br>DETECTOR</div>
                 </div>
             </div>
             <div class="hero-subtitle">{T['subtitle']}</div>
             <span class="badge">Logistic Regression</span>
             <span class="badge">Domain Analysis</span>
             <span class="badge">Explainable AI</span>
             <span class="badge">Risk Report</span>
         </div>
         <div class="hero-panel">
             <div class="hero-panel-title">Prototype readiness</div>
