import sys
import glob

inp = sys.argv[1]
out = sys.argv[2]

template = """<html>
<head>
<title></title>
</head>
<body bgcolor="white">
{0}
</body>
</html>"""

html = ""
start_ind = 0
for finp in sorted(glob.glob("%s*" % inp)):
  print(finp)
  with open(finp, 'r') as fin:
    lines = fin.read().strip().split('\n')
    for k in xrange(len(lines)):
      line = lines[k].replace("<unk>", "UNK")
      assert(line.startswith("output="))
      line = line[7:]
      # NOTE Perl script breaks if you tweak even the whitespace here...
      html += "<a name=\"{0}\">[{0}]</a> <a href=\"#{0}\" id={0}>{1}</a>\n".format(start_ind+k+1, line)
    start_ind = k

output = template.format(html.strip())
with open(out, 'w') as fout:
  fout.write(output)
