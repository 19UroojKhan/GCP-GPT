import streamlit as st
import streamlit_authenticator as stauth


def require_login():
    """Render login form and enforce authentication.

    Returns:
        (bool, Optional[stauth.Authenticate]): (authenticated, authenticator)
    """
    # Credentials (for demo/testing). Replace with secure storage for production.
    config = {
        "credentials": {
            "usernames": {
                "csadmin": {
                    "name": "csadmin",
                    "password": "u*K@19#"
                },
                "rbriggs": {
                    "name": "Rebecca Briggs",
                    "password": "def"
                }
            }
        },
        "cookie": {
            "name": "auth_cookie",
            "key": "abcdef",
            "expiry_days": 1
        }
    }

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")

    if auth_status:
        st.sidebar.success(f"Welcome {st.session_state.get('name', '')}!")
        authenticator.logout("Logout", "sidebar")
        return True, authenticator
    elif auth_status is False:
        st.error("Username/password is incorrect")
        return False, None
    else:
        st.warning("Please enter your username and password")
        return False, None


