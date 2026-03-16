# Microsoft Single Sign on einrichten

Article Description: Benutzer können sich in Poool ganz einfach über Microsoft Single Sign on Anmelden und Registrieren. Ein paar Einstellungen müssen aber vorher in Poool gemacht werden. 
Published: Yes
Suggested: No

## 📊 Artikel-Metadaten

| Artikel-Typ | Anleitung |
| --- | --- |
| Erfahrungslevel | 🚀 Grundlagen |

## Login mit Microsoft Single Sign On (SSO)

Benutzer können sich in Poool ganz einfach über **Microsoft Single Sign-On (SSO)** anmelden und registrieren. Die Grundvoraussetzung dafür ist ein aktives Microsoft 365-Konto. 

## **📌 Prozessbeschreibung: Schritt-für-Schritt Anleitung**

### **Schritt 1:** Aktivieren von Single-Sign-On in der Microsoft-Konsole

💡 *Hinweis:* Wenden Sie sich an Ihre interne IT oder Ihren **Microsoft-Administrator**, um diese Freigaben zu prüfen und ggf. zu aktivieren.

### **Schritt 2:** Login mit Microsoft Single Sign-On aktivieren

Navigieren Sie zu **`Benutzer - Login & Registrierung`** und aktivieren Sie Single-Sign-On.

![image.png](Microsoft%20Single%20Sign%20on%20einrichten/image.png)

### **Schritt 3:** Einstellungen in Poool vornehmen

- **Single Sign-On Dropdown**
Steuert, ob SSO optional, deaktiviert oder verpflichtend ist.
    - *Optional:* Anmeldung über SSO oder Benutzername/Passwort möglich.
    - *Deaktiviert:* Keine Anmeldung über SSO.
    - *Erforderlich:* Anmeldung nur über SSO erlaubt.
- **Microsoft Profilbild übernehmen**
Legt fest, ob das Microsoft-Profilbild automatisch in Poool übernommen wird.
- **Microsoft Profilsprache übernehmen**
Übernimmt automatisch die Benutzersprache aus dem Microsoft-Profil.

### **Schritt 4:** Registrierung und Benutzereinladung konfigurieren

Diese Einstellungen definieren, wie sich neue Benutzer über SSO registrieren und welche Rechte sie dabei erhalten.

- **Standard Benutzerfunktion**	
Hier können sie eine Standard Aktivität hinterlegen die neue Benutzer erhalten die sich über SSO in Poool registrieren.
- **Standard Benutzerrolle**
Eine Standard Rolle kann hier definiert werden für neue Benutzer die sich über SSO registrieren.
- **Benutzer dürfen sich via SSO registrieren**
Aktivieren Sie diese Checkbox, wenn sich Benutzer mit einem Microsoft-Konto direkt in Poool registrieren dürfen.
- **Standard Instanz**
Definiert, auf welche Instanz neue Benutzer nach der Registrierung geleitet werden (relevant bei mehreren Instanzen).
- Erlaube E-Mail Domains
Hier müssen Ihre Unternehmensdomains hinterlegt werden (z. B. `@firma.at`). Nur Benutzer dieser Domains dürfen sich via SSO registrieren.

### **Schritt 5:** Anmelden mit SSO

Nach erfolgreicher Konfiguration können sich Benutzer über den Button **`Login mit Microsoft`**anmelden.

![Untitled](Microsoft%20Single%20Sign%20on%20einrichten/Untitled.png)

## 🔍 **Zusammenfassung des Prozesses**

Über Microsoft Single Sign-On ermöglichen Sie einheitliche und sichere Logins in Poool.

Nach Aktivierung in Microsoft und Konfiguration in **Poool → Benutzer → Login & Registrierung** können sich Benutzer bequem mit ihrem Microsoft-Konto anmelden – ohne zusätzlichen Administrationsaufwand.