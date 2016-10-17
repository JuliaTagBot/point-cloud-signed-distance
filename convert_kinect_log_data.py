from __future__ import division
import sys
import lcm
import kinect
import bot_core


lc = lcm.LCM()


def convert_log(src, dest):
    srclog = lcm.EventLog(src)
    destlog = lcm.EventLog(dest, "w", overwrite=True)
    for event in srclog:
        if event.channel == "KINECT_POINTS_REDUCED":
            msg = kinect.pointcloud_t.decode(event.data)
            core_msg = bot_core.pointcloud_t()
            core_msg.utime = msg.timestamp
            core_msg.n_points = msg.num // 2
            core_msg.points = [[msg.x[i], msg.y[i], msg.z[i]]
                               for i in range(0, msg.num, 2)]
            core_msg.n_channels = 3
            core_msg.channel_names = ["r", "g", "b"]
            core_msg.channels = [[v[i] for i in range(1, msg.num, 2)]
                                 for v in [msg.x, msg.y, msg.z]]
            data = core_msg.encode()
        else:
            data = event.data
        destlog.write_event(event.timestamp, event.channel, data)
    srclog.close()
    destlog.close()




if __name__ == '__main__':
    convert_log(sys.argv[1], sys.argv[2])
