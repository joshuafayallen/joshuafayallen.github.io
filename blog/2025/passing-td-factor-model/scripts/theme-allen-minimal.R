#' A [ggplot2] that fits my aesthic preferences
#'
#'
#' @md
#' @section Why Assistant?:
#' Well for the most part I think it looks good with my LaTeX font of choice  .
#'
#'
#'
#' @md
#' @param base_family,base_size base font family and size
#' @param guide_in_plot logical to move legend inside plot. Default is FALSE
#' @param leg_pos a character vector that can be set to left or right. Default is NULL
#' @import ggplot2
#' @examples \dontrun{
#' library(ggplot2)
#' library(dplyr)
#'
#' # seminal scatterplot
#' ggplot(mtcars, aes(mpg, wt)) +
#'   geom_point() +
#'   labs(x="Fuel efficiency (mpg)", y="Weight (tons)",
#'        title="Seminal ggplot2 scatterplot example",
#'        subtitle="A plot that is only useful for demonstration purposes",
#'        caption="Brought to you by the letter 'g'") +
#'   theme_allen_minimal()
#'
#' }
#' @export
theme_allen_minimal <- function(
  base_family = "Assistant",
  base_size = 10,
  leg_pos = NULL,
  guide_in_plot = FALSE
) {
  if (guide_in_plot == TRUE && leg_pos == "right") {
    ggplot2::theme_minimal(base_family = base_family, base_size = base_size) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(
          # Font
          face = "bold",
          size = ggplot2::rel(1.285),
          colour = "#454545",
          # Center title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 8, unit = "pt")
        ),
        plot.subtitle = ggplot2::element_text(
          # Font
          family = base_family,
          face = "italic",
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Center subtitle
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 16, unit = "pt")
        ),
        plot.title.position = "plot",

        ## Caption -------------------------------------------------------------
        plot.caption = ggplot2::element_text(
          # Font
          size = ggplot2::rel(0.72),
          colour = "#454545",
          # Right-align caption
          hjust = 1,
          # Margins
          margin = ggplot2::margin(t = 20)
        ),
        plot.caption.position = "plot",

        ## Axis ----------------------------------------------------------------
        # Axis title
        axis.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1.285),
          colour = "#454545"
        ),
        # Axis Title x/y
        axis.title.y = ggplot2::element_text(
          # Right-align y axis title
          hjust = 1,
          # Margins
          margin = ggplot2::margin(r = 10)
        ),
        axis.title.x = ggplot2::element_text(
          # Left-align x axis title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(t = 10)
        ),
        # Axis labels
        axis.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1),
          colour = "#212121"
        ),
        # Axis Lines
        panel.grid = ggplot2::element_line(
          colour = "grey92"
        ),

        ## Legend -------------------------------------------------------------
        # Legend title
        legend.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545"
        ),
        # Legend labels
        legend.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.72),
          colour = "#454545"
        ),

        ## Facet Wrap ----------------------------------------------------------
        strip.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Margin
          margin = ggplot2::margin(t = 10, b = 10)
        ),
        strip.background = ggplot2::element_rect(
          fill = "white",
          colour = "transparent",
          linewidth = 0,
        ),
        panel.grid.major = element_line(
          linetype = "solid",
          color = "grey70",
          linewidth = 0.1
        ),
        panel.grid.minor = element_blank(),
        legend.position = "inside",
        legend.position.inside = c(.96, .975),
        legend.justification = c("right", "top")
      )
  } else if (leg_pos == "left" && guide_in_plot == TRUE) {
    ggplot2::theme_minimal(base_family = base_family, base_size = base_size) +
      theme(
        plot.title = ggplot2::element_text(
          # Font
          face = "bold",
          size = ggplot2::rel(1.285),
          colour = "#454545",
          # Center title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 8, unit = "pt")
        ),
        plot.subtitle = ggplot2::element_text(
          # Font
          family = base_family,
          face = "italic",
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Center subtitle
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 16, unit = "pt")
        ),
        plot.title.position = "plot",

        ## Caption -------------------------------------------------------------
        plot.caption = ggplot2::element_text(
          # Font
          size = ggplot2::rel(0.72),
          colour = "#454545",
          # Right-align caption
          hjust = 1,
          # Margins
          margin = ggplot2::margin(t = 20)
        ),
        plot.caption.position = "plot",

        ## Axis ----------------------------------------------------------------
        # Axis title
        axis.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1.285),
          colour = "#454545"
        ),
        # Axis Title x/y
        axis.title.y = ggplot2::element_text(
          # Right-align y axis title
          hjust = 1,
          # Margins
          margin = ggplot2::margin(r = 10)
        ),
        axis.title.x = ggplot2::element_text(
          # Left-align x axis title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(t = 10)
        ),
        # Axis labels
        axis.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1),
          colour = "#212121"
        ),
        # Axis Lines
        panel.grid = ggplot2::element_line(
          colour = "grey92"
        ),

        ## Legend -------------------------------------------------------------
        # Legend title
        legend.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545"
        ),
        # Legend labels
        legend.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.72),
          colour = "#454545"
        ),

        ## Facet Wrap ----------------------------------------------------------
        strip.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Margin
          margin = ggplot2::margin(t = 10, b = 10)
        ),
        strip.background = ggplot2::element_rect(
          fill = "white",
          colour = "transparent",
          linewidth = 0,
        ),
        panel.grid.major = element_line(
          linetype = "solid",
          color = "grey70",
          linewidth = 0.1
        ),
        panel.grid.minor = element_blank(),
        legend.position = "inside",
        legend.position.inside = c(.01, .89),
        legend.justification = c("left", "top")
      )
  } else {
    ggplot2::theme_minimal(base_family = base_family, base_size = base_size) +
      theme(
        plot.title = ggplot2::element_text(
          # Font
          face = "bold",
          size = ggplot2::rel(1.285),
          colour = "#454545",
          # Center title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 8, unit = "pt")
        ),
        plot.subtitle = ggplot2::element_text(
          # Font
          family = base_family,
          face = "italic",
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Center subtitle
          hjust = 0,
          # Margins
          margin = ggplot2::margin(b = 16, unit = "pt")
        ),
        plot.title.position = "plot",

        ## Caption -------------------------------------------------------------
        plot.caption = ggplot2::element_text(
          # Font
          size = ggplot2::rel(0.72),
          colour = "#454545",
          # Right-align caption
          hjust = 1,
          # Margins
          margin = ggplot2::margin(t = 20)
        ),
        plot.caption.position = "plot",

        ## Axis ----------------------------------------------------------------
        # Axis title
        axis.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1.285),
          colour = "#454545"
        ),
        # Axis Title x/y
        axis.title.y = ggplot2::element_text(
          # Right-align y axis title
          hjust = 1,
          # Margins
          margin = ggplot2::margin(r = 10)
        ),
        axis.title.x = ggplot2::element_text(
          # Left-align x axis title
          hjust = 0,
          # Margins
          margin = ggplot2::margin(t = 10)
        ),
        # Axis labels
        axis.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(1),
          colour = "#212121"
        ),
        # Axis Lines
        panel.grid = ggplot2::element_line(
          colour = "grey92"
        ),

        ## Legend -------------------------------------------------------------
        # Legend title
        legend.title = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545"
        ),
        # Legend labels
        legend.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.72),
          colour = "#454545"
        ),

        ## Facet Wrap ----------------------------------------------------------
        strip.text = ggplot2::element_text(
          # Font
          size = ggplot2::rel(.86),
          colour = "#454545",
          # Margin
          margin = ggplot2::margin(t = 10, b = 10)
        ),
        strip.background = ggplot2::element_rect(
          fill = "white",
          colour = "transparent",
          linewidth = 0,
        ),
        panel.grid.major = element_line(
          linetype = "solid",
          color = "grey70",
          linewidth = 0.1
        ),
        panel.grid.minor = element_blank()
      )
  }
}

#' Update matching font defaults for text geoms
#'
#' Updates [ggplot2::geom_label] and [ggplot2::geom_text] font defaults
#'
#' @param family,face,size,color font family name, face, size and color
#' @export
update_geom_font_defaults <- function(
  family = "Assistant",
  face = "plain",
  size = 3.5,
  color = "#2b2b2b"
) {
  update_geom_defaults(
    "text",
    list(family = family, face = face, size = size, color = color)
  )
  update_geom_defaults(
    "label",
    list(family = family, face = face, size = size, color = color)
  )
}

#' @rdname Assistant
#' @md
#' @title Assistant font name R variable aliases
#' @description `font_an` == "`Assistant`"
#' @format length 1 character vector
#' @export
font_as <- "Assistant"
