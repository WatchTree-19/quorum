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

## wdi_history.json (added 13 September 2026)

Annual World Bank WDI series for the 27 boundary sovereigns in history_raw,
1994 to 2026, fetched from the World Bank indicator API on 13 September 2026.
Three indicators are kept:

    gdp_pc_ppp     NY.GDP.PCAP.PP.KD    GDP per capita, PPP, constant 2021 intl $
    inflation      FP.CPI.TOTL.ZG       CPI inflation, annual percent
    current_acct   BN.CAB.XOKA.GD.ZS    Current account balance, percent of GDP

Coverage over 1995 to 2025 is 100.0 percent of country-years for GDP per
capita, 99.3 percent for inflation and 96.4 percent for the current account.
General government debt (GC.DOD.TOTL.GD.ZS) was fetched and then DROPPED: its
median coverage was 3 of 32 years, so its missingness would itself have been
informative about which country an item was, which defeats the blind condition.

These series exist so that the history panel can be run BLIND. Without
fundamentals a blind prompt would carry no information at all, and only the
named condition would be possible.

A quarter takes its calendar year's annual value. This is a deliberate
simplification: the agencies act on data that arrives through the year, so a
quarterly item is matched to the year's outturn rather than to what was known
at the time. It is stated here rather than hidden because it makes the task
slightly easier than the real-time problem an analyst faced.
