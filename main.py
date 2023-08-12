import os

cmd = "wget http://www.atarimania.com/roms/Roms.rar"

val = os.system(cmd)

print("exit with ", val)
