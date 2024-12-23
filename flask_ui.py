# from flask import Flask, render_template_string

# app = Flask(__name__)

# @app.route('/view-wms-map', methods=['GET'])
# def view_wms_map():
#     html_content = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Interactive WMS Map</title>
#         <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ol@latest/ol.css" />
#         <script src="https://cdn.jsdelivr.net/npm/ol@latest/ol.js"></script>
#         <style>
#             #map {
#                 width: 100%;
#                 height: 100vh;
#             }
#         </style>
#     </head>
#     <body>
#         <div id="map"></div>
#         <script>
#             // Initialize the map
#             const map = new ol.Map({
#                 target: 'map',
#                 layers: [
#                     new ol.layer.Tile({
#                         source: new ol.source.TileWMS({
#                             url: 'http://localhost:8080/geoserver/Sample_Visible/wms',
#                             params: {
#                                 'SERVICE': 'WMS',
#                                 'VERSION': '1.1.0',
#                                 'REQUEST': 'GetMap',
#                                 'LAYERS': 'Sample_Visible:3RIMG_14OCT2024_0015_L1B_STD_V01R00_IMG_VIS',
#                                 'TILED': true
#                             },
#                             serverType: 'geoserver'
#                         })
#                     })
#                 ],
#                 view: new ol.View({
#                     projection: 'EPSG:4326',
#                     center: [40, 0], // Center of the globe [longitude, latitude]
#                     zoom: 2          // Adjust zoom level
#                 })
#             });
#         </script>
#     </body>
#     </html>
#     """
#     return render_template_string(html_content)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('video_final.html')  # This will serve the HTML file

if __name__ == '__main__':
    app.run(debug=True)

