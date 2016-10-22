import os
import sys
import glob
from subprocess import check_call
from os.path import join as pjoin

# Reference scoring settings:
# https://github.com/facebook/NAMAS/blob/master/DUC/eval.sh

ROUGE_SCORER = "/home/zxie/libs/ROUGE-1.5.5/ROUGE-1.5.5.pl"

ref = sys.argv[1]  # ground truth references (glob)
dec = sys.argv[2]  # output from decoder (glob)
out = sys.argv[3]  # output directory to put processed/ROUGE inputs in

# Read decoder outputs in

def read_and_process_outputs(pattern, ref=False):
  # Process raw decoder outputs a bit
  all_lines = list()
  for f in sorted(glob.glob("%s*" % pattern)):
    print("%s" % f)
    with open(f, 'r') as fin:
      lines = fin.read().strip().split('\n')
      for k in xrange(len(lines)):
        line = lines[k]
        if not ref:
          line = line.replace("<unk>", "").strip()
        assert "<" not in line and ">" not in line, line
        assert(line.startswith("output="))
        line = line[7:].strip().replace("  ", " ")
        #line = ' '.join(tokenizer.tokenize(line))
        all_lines.append(line)
  return all_lines

ref_lines = read_and_process_outputs(ref, ref=True)
dec_lines = read_and_process_outputs(dec)

# Output in directory structure that ROUGE preparation script expects
# Requires 1 headline per file/directory...

gold_dir = pjoin(out, "gold")
system_dir = pjoin(out, "system/textsum")  # We only score 1 model
if not os.path.exists(out):
  os.makedirs(out)
  os.makedirs(gold_dir)
  os.makedirs(system_dir)
  for k in xrange(len(ref_lines)):  # Index of article
    gd = pjoin(gold_dir, "%d" % (k+1))
    # We only use 1 gold reference per but scoring script uses directories since
    # they sometimes assume multiple gold references
    os.mkdir(gd)
    with open(pjoin(gd, "%d.1.gold" % (k+1)), 'w') as fout:
      fout.write(ref_lines[k])
    with open(pjoin(system_dir, "%d.textsum.system" % (k+1)), 'w') as fout:
      fout.write(dec_lines[k])

rouge_dir = pjoin(out, "rouge")
cmd = "perl prepare4rouge-simple.pl %s %s %s" %\
        (rouge_dir, os.path.dirname(system_dir), gold_dir)
print(cmd)
check_call(cmd, shell=True)

# Finally, score (limited length to 75 bytes as in Rush paper)

cmd = "cd %s; %s -m -b 75 -n 2 -e %s -a settings.xml; cd -" %\
        (rouge_dir, ROUGE_SCORER, os.path.dirname(ROUGE_SCORER) + "/data")
print(cmd)
check_call(cmd, shell=True)
