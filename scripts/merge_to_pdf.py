import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
from svglib import svglib
from reportlab.graphics.shapes import Rect


# Function to add an image to a PDF
def add_image_to_pdf(pdf_canvas, image_path, x, y, width, height):
    pdf_canvas.drawImage(ImageReader(image_path), x, -y - height,
                         width, height, preserveAspectRatio=True)


def add_svg_to_pdf(pdf_canvas, svg_path, x, y, width, height, cropping=[0, 0, 0, 0]):
    """The format from cropping is a list of [x_crop_from_left, x_crop_from_right,
        y_crop_from_top, y_crop_from_bottom]. The values are in pixels."""
    drawing = svglib.svg2rlg(svg_path)
    drawing.scale(width/drawing.width, height/drawing.height)
    drawing.width = width
    drawing.height = height
    if cropping == [0, 0, 0, 0]:
        renderPDF.draw(drawing, pdf_canvas, x, -y - height)
        return
    # mask the cropped regions
    x_crop_from_left, x_crop_from_right, y_crop_from_top, y_crop_from_bottom = cropping
    drawing.add(Rect(0, 0, x_crop_from_left * 2,
                height * 2, fillColor=(1, 1, 1), strokeColor=(1, 1, 1)))
    drawing.add(Rect((width - x_crop_from_right) * 2,
                0, x_crop_from_right * 2, height * 2, fillColor=(1, 1, 1), strokeColor=(1, 1, 1)))
    drawing.add(
        Rect(0, 0, width * 2, y_crop_from_top * 2, fillColor=(1, 1, 1), strokeColor=(1, 1, 1)))
    drawing.add(Rect(0, (height - y_crop_from_bottom) * 2,
                width * 2, y_crop_from_bottom * 2, fillColor=(1, 1, 1), strokeColor=(1, 1, 1)))
    renderPDF.draw(drawing, pdf_canvas, x - x_crop_from_left, -
                   y - height + y_crop_from_bottom)


def create_red_dwarf_plot():
    # Set the page size dynamically based on plot dimensions
    plot1_x, plot1_y, plot1_width, plot1_height = 0, 0, 400, 300
    plot2_x, plot2_y, plot2_width, plot2_height = 2, 300, 400, 300
    spectra_width = 400
    spectra_height = 230
    plot3_x, plot3_y, plot3_width, plot3_height = 420, 150, spectra_width, spectra_height
    plot4_x, plot4_y, plot4_width, plot4_height = 420, plot3_y + spectra_height - \
        50, spectra_width, spectra_height

    max_width = max([plot1_x + plot1_width, plot2_x + plot2_width, plot3_x +
                    plot3_width, plot4_x + plot4_width])
    max_height = max([plot1_y + plot1_height, plot2_y + plot2_height, plot3_y +
                      plot3_height, plot4_y + plot4_height])

    # Create a new PDF with dynamically set page size
    pdf_path = "combined_figure_red_dwarfs.pdf"
    pdf = canvas.Canvas(pdf_path, pagesize=(max_width, max_height))
    pdf.translate(0, max_height)
    # Plot HR_diagram.png
    add_image_to_pdf(pdf, "umap_images/(a)_CMD_diagram_-_red_dwarfs_with_title.png",
                     plot1_x, plot1_y, plot1_width, plot1_height)

    # Plot UMAP.png
    add_image_to_pdf(pdf, "umap_images/(b)_UMAP_-_red_dwarfs_with_title.png",
                     plot2_x, plot2_y, plot2_width, plot2_height)

    # Add arrows connecting highlighted cluster in HR_diagram to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # Arrow from HR_diagram to UMAP cluster 1
    clust_loc_in_HR = [165 + plot1_x, 250 + plot1_y]

    # Arrow from HR_diagram to UMAP cluster 2
    clust2_loc_in_UMAP = [236 + plot2_x, 185 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust2_loc_in_UMAP[0],
             -clust2_loc_in_UMAP[1])

    # Arrow from HR_diagram to UMAP cluster 3
    clust3_loc_in_UMAP = [220 + plot2_x, 243 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust3_loc_in_UMAP[0],
             -clust3_loc_in_UMAP[1])

    # Plot spect1.svg
    add_svg_to_pdf(pdf, "umap_images/red_dwarfs_main_clust_spect.svg",
                   plot3_x, plot3_y, plot3_width, plot3_height)

    # Plot spect2.svg
    add_svg_to_pdf(pdf, "umap_images/red_dwarfs_separate_clust_spect.svg",
                   plot4_x, plot4_y, plot4_width, plot4_height)

    # add lines from spectra to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # arrow from spectra to UMAP cluster 2
    clust2_loc_in_spect = [plot3_x - 5, spectra_height / 2 + plot3_y]
    pdf.line(clust2_loc_in_spect[0], -clust2_loc_in_spect[1], clust2_loc_in_UMAP[0] + 15,
             -clust2_loc_in_UMAP[1])

    # arrow from spectra to UMAP cluster 3
    clust3_loc_in_spect = [plot4_x - 5, spectra_height / 2 + plot4_y]
    pdf.line(clust3_loc_in_spect[0], -clust3_loc_in_spect[1], clust3_loc_in_UMAP[0] + 15,
             -clust3_loc_in_UMAP[1])
    # Save the PDF
    pdf.save()

    print(f"Combined figure saved to {pdf_path}")


def create_blue_giant_plot():
    # Set the page size dynamically based on plot dimensions
    plot1_x, plot1_y, plot1_width, plot1_height = 0, 0, 400, 300
    plot2_x, plot2_y, plot2_width, plot2_height = 2, 300, 400, 300
    spectra_width = 400
    spectra_height = 230
    plot3_x, plot3_y, plot3_width, plot3_height = 420, 0, spectra_width, spectra_height
    plot4_x, plot4_y, plot4_width, plot4_height = 420, plot3_y + spectra_height - \
        50, spectra_width, spectra_height
    plot5_x, plot5_y, plot5_width, plot5_height = 420, plot4_y + \
        spectra_height - 50, spectra_width, spectra_height

    max_width = max([plot1_x + plot1_width, plot2_x + plot2_width, plot3_x +
                    plot3_width, plot4_x + plot4_width, plot5_x + plot5_width])
    max_height = max([plot1_y + plot1_height, plot2_y + plot2_height, plot3_y +
                      plot3_height, plot4_y + plot4_height, plot5_y + plot5_height])

    # Create a new PDF with dynamically set page size
    pdf_path = "combined_figure_blue_giants.pdf"
    pdf = canvas.Canvas(pdf_path, pagesize=(max_width, max_height))
    pdf.translate(0, max_height)
    # Plot HR_diagram.png
    add_image_to_pdf(pdf, "umap_images/(a)_CMD_diagram_-_blue_giants_with_title.png",
                     plot1_x, plot1_y, plot1_width, plot1_height)

    # Plot UMAP.png
    add_image_to_pdf(pdf, "umap_images/(b)_UMAP_-_blue_giants_with_title.png",
                     plot2_x, plot2_y, plot2_width, plot2_height)

    # Add arrows connecting highlighted cluster in HR_diagram to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # Arrow from HR_diagram to UMAP cluster 1
    clust_loc_in_HR = [90 + plot1_x, 180 + plot1_y]
    clust1_loc_in_UMAP = [250 + plot2_x, 180 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust1_loc_in_UMAP[0],
             -clust1_loc_in_UMAP[1])

    # Arrow from HR_diagram to UMAP cluster 2
    clust2_loc_in_UMAP = [280 + plot2_x, 50 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust2_loc_in_UMAP[0],
             -clust2_loc_in_UMAP[1])

    # Arrow from HR_diagram to UMAP cluster 3
    clust3_loc_in_UMAP = [280 + plot2_x, 120 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust3_loc_in_UMAP[0],
             -clust3_loc_in_UMAP[1])

    # Plot spect1.svg
    add_svg_to_pdf(pdf, "umap_images/blue_giants_separate_group_spect.svg",
                   plot3_x, plot3_y, plot3_width, plot3_height)

    # Plot spect2.svg
    add_svg_to_pdf(pdf, "umap_images/blue_giants_intermediate_group_spect.svg",
                   plot4_x, plot4_y, plot4_width, plot4_height)

    # Plot spect3.svg
    add_svg_to_pdf(pdf, "umap_images/blue_giants_main_clust_spect.svg",
                   plot5_x, plot5_y, plot5_width, plot5_height)

    # add lines from spectra to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # arrow from spectra to UMAP cluster 1
    clust1_loc_in_spect = [plot5_x - 5, spectra_height / 2 + plot5_y]
    pdf.line(clust1_loc_in_spect[0], -clust1_loc_in_spect[1], clust1_loc_in_UMAP[0] + 30,
             -clust1_loc_in_UMAP[1] - 20)

    # arrow from spectra to UMAP cluster 2
    clust2_loc_in_spect = [plot3_x - 5, spectra_height / 2 + plot3_y]
    pdf.line(clust2_loc_in_spect[0], -clust2_loc_in_spect[1], clust2_loc_in_UMAP[0] + 15,
             -clust2_loc_in_UMAP[1])

    # arrow from spectra to UMAP cluster 3
    clust3_loc_in_spect = [plot4_x - 5, spectra_height / 2 + plot4_y]
    pdf.line(clust3_loc_in_spect[0], -clust3_loc_in_spect[1], clust3_loc_in_UMAP[0] + 15,
             -clust3_loc_in_UMAP[1])
    # Save the PDF
    pdf.save()

    print(f"Combined figure saved to {pdf_path}")


def create_red_supergiant_plot():
    # Set the page size dynamically based on plot dimensions
    plot1_x, plot1_y, plot1_width, plot1_height = 0, 0, 400, 300
    plot2_x, plot2_y, plot2_width, plot2_height = 2, 300, 400, 300
    spectra_width = 400
    spectra_height = 230
    plot3_x, plot3_y, plot3_width, plot3_height = 420, 190, spectra_width, spectra_height
    plot4_x, plot4_y, plot4_width, plot4_height = 420, plot3_y + spectra_height - \
        50, spectra_width, spectra_height

    max_width = max([plot1_x + plot1_width, plot2_x + plot2_width, plot3_x +
                    plot3_width, plot4_x + plot4_width])
    max_height = max([plot1_y + plot1_height, plot2_y + plot2_height, plot3_y +
                      plot3_height, plot4_y + plot4_height])

    # Create a new PDF with dynamically set page size
    pdf_path = "combined_figure_red_supergiants.pdf"
    pdf = canvas.Canvas(pdf_path, pagesize=(max_width, max_height))
    pdf.translate(0, max_height)
    # Plot HR_diagram.png
    add_image_to_pdf(pdf, "umap_images/(a)_CMD_diagram_-_red_supergiants_with_title.png",
                     plot1_x, plot1_y, plot1_width, plot1_height)

    # Plot UMAP.png
    add_image_to_pdf(pdf, "umap_images/(b)_UMAP_-_red_supergiants_with_title.png",
                     plot2_x, plot2_y, plot2_width, plot2_height)

    # Add arrows connecting highlighted cluster in HR_diagram to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # Arrow from HR_diagram to UMAP cluster 1
    clust_loc_in_HR = [190 + plot1_x, 150 + plot1_y]

    # Arrow from HR_diagram to UMAP cluster 2
    clust2_loc_in_UMAP = [205 + plot2_x, 130 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust2_loc_in_UMAP[0]+10,
             -clust2_loc_in_UMAP[1]+25)

    # Arrow from HR_diagram to UMAP cluster 3
    clust3_loc_in_UMAP = [50 + plot2_x, 125 + plot2_y]
    pdf.line(clust_loc_in_HR[0], -clust_loc_in_HR[1], clust3_loc_in_UMAP[0]+20,
             -clust3_loc_in_UMAP[1]+35)

    # Plot spect1.svg
    add_svg_to_pdf(pdf, "umap_images/red_supergiants_main_clust_spect.svg",
                   plot3_x, plot3_y, plot3_width, plot3_height)

    # Plot spect2.svg
    add_svg_to_pdf(pdf, "umap_images/red_supergiants_separate_clust_spect.svg",
                   plot4_x, plot4_y, plot4_width, plot4_height)

    # add lines from spectra to UMAP clusters
    pdf.setLineWidth(0.8)
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf._setStrokeAlpha(0.4)

    # arrow from spectra to UMAP cluster 2
    clust2_loc_in_spect = [plot3_x - 5, spectra_height / 2 + plot3_y]
    pdf.line(clust2_loc_in_spect[0], -clust2_loc_in_spect[1], clust2_loc_in_UMAP[0] + 30,
             -clust2_loc_in_UMAP[1])

    # arrow from spectra to UMAP cluster 3
    clust3_loc_in_spect = [plot4_x - 5, spectra_height / 2 + plot4_y]
    pdf.line(clust3_loc_in_spect[0], -clust3_loc_in_spect[1], clust3_loc_in_UMAP[0] + 30,
             -clust3_loc_in_UMAP[1])
    # Save the PDF
    pdf.save()

    print(f"Combined figure saved to {pdf_path}")


if __name__ == "__main__":
    # create_blue_giant_plot()
    # create_red_dwarf_plot()
    create_red_supergiant_plot()
