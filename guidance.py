def get_guidance(category):
    if category in ["sexual", "physical", "threat", "stalking"]:
        return {
            "law": "IPC 354, 354A, 506 | IT Act 66E",
            "tips": [
                "Save messages/screenshots",
                "Note dates, time, location",
                "Do not delete evidence"
            ],
            "sos": ["112 (Emergency)", "1091 (Women Helpline)"]
        }
    return {
        "message": "Your experience matters. If this felt uncomfortable, you are allowed to set boundaries."
    }
