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

# FIX LAT/LON
if ("lat" %in% names(cyclone)) cyclone$latitude <- cyclone$lat
if ("lon" %in% names(cyclone)) cyclone$longitude <- cyclone$lon

# CLEAN DATA
cyclone <- cyclone %>%
  mutate(
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude)
  ) %>%
  filter(!is.na(latitude), !is.na(longitude))

# LOAD MAP
world <- map_data("world")

# ---------------------------
# HOTSPOT MAP (CURVED - NO BOX)
# ---------------------------
ggplot() +
  geom_polygon(
    data = world,
    aes(long, lat, group = group),
    fill = "gray90",
    color = "gray50"
  ) +
  stat_density_2d(
    data = cyclone,
    aes(longitude, latitude, fill = after_stat(level)),
    geom = "polygon",
    alpha = 0.6
  ) +
  scale_fill_viridis_c(option = "plasma") +
  
  # 🔥 THIS REMOVES RECTANGLE SHAPE
  coord_map("ortho", orientation = c(-20, 50, 0)) +
  
  theme_void() +
  labs(
    title = "Cyclone Hotspots (Curved Earth View)",
    fill = "Density"
  )
