# Sovereign panel: data provenance

sovereign_panel.csv, built September 2026. Fully public sources; nothing
proprietary.

Ratings (columns sp, moodys, fitch): long-term foreign-currency sovereign
issuer ratings as publicly compiled at retrieval time (Wikipedia, "List of
countries by credit rating", retrieved 2026-09-12). Countries rated by fewer
than two of the three agencies are excluded by the adapter, per the schema.
Before any formal publication, re-verify each rating against the agencies'
own public rating actions; a compiled tertiary source is fine for a pilot
and not for a release.

Fundamentals (columns gdp_pc_usd, inflation_pct, current_account_pct_gdp):
World Bank World Development Indicators API, most recent non-empty value per
country at retrieval time (indicators NY.GDP.PCAP.CD, FP.CPI.TOTL.ZG,
BN.CAB.XOKA.GD.ZS). The raw fetched values are kept verbatim in
wb_gdppc.txt, wb_inflation.txt, wb_current_account.txt.

sovereign_ratings.csv is the ratings-only intermediate; sovereign_panel.csv
is the merged file the adapter reads.

History panel (data/history_raw/<ISO3>.txt): dated long-term foreign-currency
rating actions per agency for 27 boundary sovereigns, compiled from
countryeconomy.com rating-history pages (retrieved 2026-09-12), affirmations
at an unchanged rating dropped. Selection is deliberate: countries that
crossed or approached the investment-grade line since the 1990s; the panel
measures behaviour at the boundary, not the base rate among all sovereigns.
Known truncations at retrieval: Serbia lacks the Moody's series; Oman's and
Azerbaijan's most recent Fitch actions may be missing. Where the history and
the cross-sectional snapshot disagree, the history page was the fresher
source. Re-verify every action against the agencies' own published rating
actions before formal release.
