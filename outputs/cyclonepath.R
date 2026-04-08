# ---------------------------
# LOAD LIBRARIES
# ---------------------------
packages <- c("ggplot2","dplyr","readr","maps","viridis")
for (p in packages) {
  if (!require(p, character.only = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

# ---------------------------
# LOAD DATA
# ---------------------------
cyclone <- read_csv("D:/final_era5.csv", show_col_types = FALSE)
names(cyclone) <- tolower(names(cyclone))

# ---------------------------
# FIX LAT/LON
# ---------------------------
if ("lat" %in% names(cyclone)) cyclone$latitude <- cyclone$lat
if ("lon" %in% names(cyclone)) cyclone$longitude <- cyclone$lon

# ---------------------------
# CREATE WIND SPEED (IMPORTANT)
# ---------------------------
if (!"wind_speed" %in% names(cyclone)) {
  if (all(c("u10","v10") %in% names(cyclone))) {
    cyclone <- cyclone %>%
      mutate(wind_speed = sqrt(u10^2 + v10^2))
  }
}

# ---------------------------
# CLEAN DATA
# ---------------------------
cyclone <- cyclone %>%
  mutate(
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude),
    wind_speed = as.numeric(wind_speed)
  ) %>%
  filter(!is.na(latitude), !is.na(longitude))

# ---------------------------
# DEFINE HEAVY CYCLONE THRESHOLD
# ---------------------------
threshold <- quantile(cyclone$wind_speed, 0.90, na.rm = TRUE)

heavy <- cyclone %>%
  filter(wind_speed >= threshold)

# ---------------------------
# CREATE TRACK GROUPS
# ---------------------------
cyclone <- cyclone %>%
  mutate(cyclone_id = cumsum(c(1,
    abs(diff(latitude)) > 2 | abs(diff(longitude)) > 2
  )))

# ---------------------------
# LOAD WORLD MAP
# ---------------------------
world <- map_data("world")

# ---------------------------
# FINAL VISUALIZATION
# ---------------------------
ggplot() +
  geom_polygon(
    data = world,
    aes(long, lat, group = group),
    fill = "gray96",
    color = "gray70",
    linewidth = 0.2
  ) +

  # 🔴 Cyclone paths
  geom_path(
    data = cyclone,
    aes(longitude, latitude, group = cyclone_id),
    color = "#ff6b6b",
    linewidth = 0.25,
    alpha = 0.6
  ) +

  # 🔥 Heavy cyclone points
  geom_point(
    data = heavy,
    aes(longitude, latitude),
    color = "red",
    size = 1.5,
    alpha = 0.9
  ) +

  coord_fixed(1.3) +
  theme_void() +

  labs(
    title = "Cyclone Paths with Heavy Cyclone Hotspots",
    subtitle = "Red points indicate high-intensity cyclone regions"
  )
