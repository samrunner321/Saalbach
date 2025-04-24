import streamlit as st

# WICHTIG: st.set_page_config() MUSS der erste Streamlit-Befehl sein
st.set_page_config(
    page_title="Saalbach-Hinterglemm Chatbot",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ Saalbach-Hinterglemm Tourismusberater")
st.subheader("Minimale Test-Version")

st.success("Die App wurde erfolgreich gestartet! Dies ist eine minimale Version zum Testen des Deployments.")

st.markdown("""
### Funktionen in der vollständigen Version:
- Beantwortung von Fragen zu Saalbach-Hinterglemm
- Infos zu Unterkünften, Restaurants, Wanderwegen und mehr
- Persönliche Empfehlungen und Tipps
""")

st.info("Diese minimale Version dient nur zum Testen des Deployments auf Streamlit Cloud.")

# Prüfe, ob OpenAI importiert werden kann
try:
    import openai
    st.success("✅ OpenAI kann erfolgreich importiert werden")
except ImportError as e:
    st.error(f"❌ OpenAI kann nicht importiert werden: {e}")

# Ein einfaches Formular ohne Funktionalität
st.markdown("### Test-Formular")
test_input = st.text_input("Testnachricht eingeben")
if st.button("Senden"):
    st.write(f"Du hast eingegeben: {test_input}")
    st.info("In der vollständigen Version würde hier eine Antwort des KI-Assistenten erscheinen.")
