pacman::p_load(
  "tidyverse",
  "MetBrewer"
  "arrow",
  "data.table",
  "ggdag",
  "dagitty",
  "patchwork",
  "posterior",
  "kableExtra",
  install = truelength()
)


shorten_dag_arrows <- function(tidy_dag, proportion){
  # Update underlying ggdag object
  tidy_dag$data <- dplyr::mutate(tidy_dag$data, 
                                 xend = (1-proportion/2)*(xend - x) + x, 
                                 yend = (1-proportion/2)*(yend - y) + y,
                                 xstart = (1-proportion/2)*(x - xend) + xend,
                                 ystart = (1-proportion/2)*(y-yend) + yend)
  return(tidy_dag)
}

coords_for_dag <-  tribble(~name, ~x, ~y,
                           "x", 0, 0,
                           "y", 4, 0,
                           "z1", 2,2,
                           "z2", 2,-2,)

dag_labels <- list(
    x = "X", y = "Y", z1 = "bold(Z[1])", 
    z2 = "bold(Z[2])")

dags_with_coords <-  dagify(y ~ x + z1 + z2,
                            x ~ z1 + z2,
                            exposure = "x",
                            outcome = "y",
                            coords = coords_for_dag,
                            labels = dag_labels) |>
                            tidy_dagitty()

simple_dag = shorten_dag_arrows(dags_with_coords,
                                proportion = 0.08) |>
            mutate(.text_color = case_when(name == "x" ~ "#262d42",
                                          name == "y" ~ "#591c19",
                                          name %in% c("z1", "z2") ~ "#808080"))



 simple_dag = ggplot(simple_dag, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(aes(color = .text_color)) +
  geom_dag_edges(aes(x = xstart, y = ystart), edge_width = 1.5,
    arrow_directed = grid::arrow(
      length = grid::unit(10, "pt"), 
      type = "closed"), show.legend = FALSE) + 
  scale_color_met_d(name = "Lakota") +
  theme_dag() +
  theme(legend.position = "none")



simple_dag

ggsave("simple_dag.svg", plot = simple_dag,  device = "svg", path = here::here("static") )


library(hexSticker)

img_dag = magick::image_read("static/simple_dag.svg")

library(showtext)

font_add_google("Jost")

showtext_auto()



sticker(
  subplot = img_dag, 
  package = "Political Science Research Methods", 
  p_size = 12, 
  p_y = 1.58,
  p_color = "#000000",
  p_family = "Jost",
  p_fontface = "bold",
  s_x = 1, 
  s_y = 0.93, 
  s_width = 1.85, 
  s_height = 1.8,
  h_fill = "#FFFFFF",
  h_color = "#0039A6",
  dpi = 600,
  filename="teaching/logos/research_methods_logo.png"
  )