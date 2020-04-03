import numpy as np
import matplotlib
import mpl_toolkits
from acrolib.plotting import get_default_axes3d, plot_reference_frame


def test_create_axes_3d():
    fig, ax = get_default_axes3d()
    assert isinstance(fig, matplotlib.pyplot.Figure)
    assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D)


def test_plot_reference_frame():
    _, ax = get_default_axes3d()
    plot_reference_frame(ax)
    plot_reference_frame(ax, tf=np.eye(4))
    plot_reference_frame(ax, tf=np.eye(4), arrow_length=0.3)
