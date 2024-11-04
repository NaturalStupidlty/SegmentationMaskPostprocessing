import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import contextily as ctx

MIN_Y, MAX_Y = 5937000, 5937500
MIN_X, MAX_X = 3561000, 3561500

polygon_coords = [
    [(3561115.70, 5937335.13), (3561109.97, 5937326.85), (3561122.71, 5937319.84), (3561125.90, 5937328.76)],
    [(3561201.73, 5937293.71), (3561201.09, 5937289.89), (3561205.55, 5937287.98), (3561208.10, 5937291.16)],
    [(3561210.65, 5937305.18), (3561213.84, 5937300.08), (3561218.93, 5937308.37), (3561213.84, 5937309.01)],
]

clicked_points = []
blue_points = []


def on_click(event):
    global clicked_points, ax, blue_points
    if event.inaxes is not None:
        if event.button == 1 and len(clicked_points) < 4:

            clicked_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'o', color='red', markersize=1)
            plt.draw()
            print(f'Left Click: ({event.xdata:.2f}, {event.ydata:.2f})')

            if len(clicked_points) == 4:
                draw_polygon(clicked_points)
                clicked_points.clear()

        elif event.button == 3:

            blue_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'o', color='blue', markersize=2)
            plt.draw()
            print(f'Right Click: ({event.xdata:.2f}, {event.ydata:.2f})')
            draw_lines_between_blue_points()

def draw_polygon(points):
    polygon = Polygon(points)
    gdf_new = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:3857")

    gdf_new.plot(ax=ax, color='orange', alpha=0.5, edgecolor="black", linewidth=0.7)


def draw_lines_between_blue_points():
    if len(blue_points) > 1:

        x_values, y_values = zip(*blue_points[-2:])
        ax.plot(x_values, y_values, color='black', linewidth=0.3)  # Draw lines between points
        plt.draw()


def clear_points(event):
    global clicked_points, blue_points
    clicked_points.clear()
    blue_points.clear()
    ax.clear()
    setup_map_display()
    plt.draw()


def setup_map_display():
    polygons = [Polygon(coords) for coords in polygon_coords]
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:3857")

    colors = ['tomato', 'steelblue', 'limegreen']
    for polygon, color in zip(gdf.geometry, colors):
        gpd.GeoSeries([polygon]).plot(ax=ax, color=color, alpha=0.5, edgecolor="black", linewidth=0.7)

    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=16)


fig, ax = plt.subplots(figsize=(15, 15))
setup_map_display()

fig.canvas.mpl_connect('key_press_event', clear_points)

cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.xlabel('Lat')
plt.ylabel('Lng')


plt.show()
