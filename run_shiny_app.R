# ── run_shiny_app.R ──────────────────────────────────────────────────────────
# Launch the TimeKAN Explorer Shiny app.
# Run from the project root:
#   Rscript run_shiny_app.R
# Then open  http://127.0.0.1:7777  in your browser.
# ─────────────────────────────────────────────────────────────────────────────
library(shiny)
runApp("shiny_timekan_explorer", port = 7777, launch.browser = TRUE)

