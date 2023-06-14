import os
import sys
script_dir = os.getcwd()
module_path = script_dir
for _ in range(5):
  module_path = os.path.abspath(os.path.join(module_path, '../'))
  if module_path not in sys.path:
    sys.path.insert(0, module_path)
  if os.path.basename(module_path) == 'MODESA':
    break
from src.unit import Unit
from src.operators import *
import src.operators
from src.operator_base import op_type_dicts
from src.system import System
import pandas as pd
from src.analye_model import *
import bokeh.plotting as plt
import bokeh.layouts as pltLayout
from bokeh.models import ColumnDataSource, DataTable, HoverTool, TableColumn


def plot_roofline_background(title, system: System, max_x) -> plt.figure:
  op_intensity = system.flops / system.offchip_mem_bw
  flops = unit.raw_to_unit(system.flops, type='C')
  max_x = max(max_x, op_intensity * 2.5)
  turning_points = [[0, 0], [op_intensity, flops], [max_x, flops]]
  turning_points = np.array(turning_points)
  fig = plt.figure(tools="pan,wheel_zoom,box_zoom,reset,save",
                   toolbar_location="above", width=1200, height=600, title=title)

  fig.line(turning_points[:, 0], turning_points[:, 1],
           legend_label="offchip bandwidth", line_color='grey')

  op_intensity = system.flops / system.onchip_mem_bw
  flops = unit.raw_to_unit(system.flops, type='C')
  turning_points = [[0, 0], [op_intensity, flops], [max_x, flops]]
  turning_points = np.array(turning_points)
  fig.line(turning_points[:, 0], turning_points[:, 1],
           legend_label="inchip bandwidth", line_dash='4 4', line_color='grey')

  fig.xaxis.axis_label = 'Op Intensity (FLOPs/Byte)'
  fig.yaxis.axis_label = f'{unit.unit_compute.upper()}'
  return fig


def dot_roofline(title: str, df) -> plt.figure:
  max_x = max(df['Op Intensity'])
  fig = plot_roofline_background(title, system, max_x)
  source = ColumnDataSource(df)

  datatable = DataTable(
      source=source,
      columns=[
          TableColumn(field="Op Type", title="Op Type"),
          TableColumn(field="Dimension", title="Dimension"),
          TableColumn(field="Bound", title="Bound"),
          TableColumn(field="C/M ratio", title="C/M ratio"),
          TableColumn(field="Op Intensity", title="Op Intensity"),
      ],
      editable=False, width=1200, height=350,
      index_position=-1, index_header="row index", index_width=60)

  points = fig.scatter(x='Op Intensity', y='Throughput (Tflops)', size=10, source=source)
  hover = HoverTool(tooltips=[
      ("Op Type", "@{Op Type}"),
      ("Dimension", "@Dimension"),
  ], renderers=[points])
  fig.tools.append(hover)
  return pltLayout.column(fig, datatable)


if __name__ == "__main__":
  model = 'resnet50'
  data_path = os.path.join(module_path, "data")

  # seq_len = 256
  # model = 'custom_attn_vanilla'
  # data_path = os.path.join(module_path,"data")
  # model_path = os.path.join(data_path,"model")
  # create_model(seq_len, name=model, data_path=data_path)

  batch_size = 1
  unit = Unit()
  system = System(unit, onchip_mem_bw=64, offchip_mem_bw=3.2,
                  on_chip_mem_size=4, off_chip_mem_size=512,
                  flops=1.2, frequency=800, compress_mem=False)
  model_df = get_model_df(model, system, unit, batch_size, data_path, sparse=False)
  print(model_df)

  fig = dot_roofline(model, model_df)
  get_summary_table(model_df)
  plt.show(fig)
