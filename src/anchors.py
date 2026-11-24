"""Leakage-free anchor set and operating point for the semantic rung.

Copied from the notebook (Week 3, Step 1B) so the app and the notebook share one
definition. These sentences are DISJOINT from everything used to generate or
evaluate proposals -- the fix for the anchor-leakage bug found in Step 1A.
"""

ANCHORS_V2 = {
    "P1_biometric":      ["Scan every visitor's face at the door and match it against our customer database.",
                          "Build facial templates of shoppers so our systems know who is who in every store."],
    "P2_data_selling":   ["Hand our user records to outside companies that pay us for access.",
                          "Turn the personal information we hold into an income source by licensing it out."],
    "P3_consent":        ["Record where people are at all times without ever telling them.",
                          "Switch tracking on for everyone unless they discover the buried setting to disable it."],
    "P4_discrimination": ["Filter out older applicants automatically so we hire only younger staff.",
                          "Rate loan applications by home district so certain communities score lower."],
    "P5_oversight":      ["The software gives the final yes or no on claims and nobody double-checks it.",
                          "Machines make the rejection decision alone, with staff kept out of the process."],
    "P6_dark_pattern":   ["Tick the sharing-consent box for the user in advance and bury the way out.",
                          "Add costs at the last payment step that customers only see once charged."],
}

# Chosen in the notebook: max adversarial recall subject to precision >= 0.60.
SEMANTIC_THRESHOLD = 0.40

# Router band: below -> auto-clear, inside -> LLM judge, above -> auto-flag.
UNCERTAIN_BAND = (0.30, 0.50)
