import os
import sys
import glob
import traceback
from matplotlib import pyplot as plt

class DatasetStatistics:

  def __init__(self, title_fontsize=14, label_fontsize=12):
    pass
    self.title_fontsize = title_fontsize
    self.label_fontsize = label_fontsize


  def setvalue(self, graph, height):
    for rect in graph:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')

  
  def plot(self, dataset_dir, ymax=1000):
    title   = os.path.basename(dataset_dir)
    subdirs = os.listdir(dataset_dir)
    subdirs = sorted(subdirs)
    counts  = []
    YMAX = 0
    for dir in subdirs:
       dir = os.path.join(dataset_dir, dir)      
       #print(" {} ".format(dir))
       files = glob.glob(dir + "/*.png")
       count = len(files)
       if count >YMAX:
        YMAX = count
       counts.append(count)
       dir = os.path.basename(dir)
       print("dir {}  count {}".format(dir, count))
    # Creating a simple bar chart
    ymax = YMAX + 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, ymax)
    #fig.tight_layout()

    graph = plt.bar(subdirs, counts)
    self.setvalue(graph, counts)
    plt.title(title)
    #plt.xlabel("Labels", fontsize=self.label_fontsize)
    ax.set_xlabel("Labels", fontsize = self.label_fontsize, weight = 'bold')
    ax.set_ylabel("Count", fontsize = self.label_fontsize, weight = 'bold')
    ax.set_title(title, fontsize = self.title_fontsize, weight = 'bold', pad = 20)

    #plt.show()
    filename = os.path.basename(dataset_dir)
    filename = dataset_dir.replace("/", "_")
    filename = filename.replace("\\", "_")
    filename = filename.replace(".", "")
    figfilename = os.path.join("./", filename + ".png")
    fig.savefig(figfilename)


# python DatasetStatistics.py ./BreaKHis_V1_400X/train
if __name__ == "__main__":
  try:
    dataset_dir = ""
    if len(sys.argv) == 2:
      dataset_dir = sys.argv[1]
    else:
      raise Exception("Invalid argment")

    stat = DatasetStatistics()
    stat.plot(dataset_dir)

  except:
    traceback.print_exc()