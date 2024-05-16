import os
import argparse
import glob
import re
import numpy as np

from plyfile import PlyData
from PIL import Image
import OpenEXR
import Imath


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    if not os.path.isfile(filename):
        print(filename)
        assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


def get_xml(args):
    xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            """ + \
    """        <float name="fov" value="{}"/>""".format(args.fov) + \
    """
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

    xml_ball_segment = \
    """
        <shape type="sphere">""" + \
    """
            <float name="radius" value="{}"/>""".format(args.radius) + \
    """
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

    return xml_head, xml_ball_segment, xml_tail


def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i]<=0.0031308,
                (rgb[i]*12.92)*255.0,
                (1.055*(rgb[i]**(1.0/2.4))-0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tobytes()).convert("L") for c in rgb]
    #rgb8 = [Image.fromarray(c.astype(int)) for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)


def EncodeToSRGB(v):
    if (v <= 0.0031308):
        return (v * 12.92) * 255.0
    else:
        return (1.055*(v**(1.0/2.4))-0.055) * 255.0


def main(args):
    if args.all:
        print("Render all ply in a folder")
        assert args.folder is not None
        folder_name = args.folder
        file_names = [f.split('/')[-1][:-4] for f in glob.glob(args.folder+'/*.ply')]
    else:
        assert args.input is not None
        folder_name = os.path.join(*args.input.split('/')[:-1])
        file_names = [args.input.split('/')[-1][:-4]]

    for file_name in file_names:
        working_folder = os.path.join(folder_name, file_name)
        if not os.path.exists(working_folder):
            os.mkdir(working_folder)
        xml_head, xml_ball_segment, xml_tail = get_xml(args)
        xml_segments = [xml_head]

        ply_file_name = os.path.join(folder_name, file_name + '.ply')
        pcl = read_ply(ply_file_name)
        # pcl = standardize_bbox(pcl, 2048)
        pcl = standardize_bbox(pcl, len(pcl))
        pcl = pcl[:,[2,0,1]]
        pcl[:,0] *= -1
        pcl[:,2] += 0.0125

        if args.color is not None:
            color = [int(c)/255.0 for c in re.findall(r'\d+', args.color)]
        for i in range(pcl.shape[0]):
            if args.color is None:
                color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xml_name = os.path.join(working_folder, file_name + '.xml')
        with open(xml_name, 'w') as f:
            f.write(xml_content)

        # render
        os.system('mitsuba {}'.format(xml_name))

        exr_name = os.path.join(working_folder, file_name + '.exr')
        im_name = os.path.join(working_folder, file_name + '.png')
        ConvertEXRToJPG(exr_name, im_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--folder', '-f', type=str, default=None)
    parser.add_argument('--input', '-i', type=str, default=None)
    parser.add_argument('--radius', '-r', type=float, default=0.008)
    parser.add_argument('--fov', type=float, default=24)
    parser.add_argument('--color', '-c', type=str, default=None)

    args = parser.parse_args()
    main(args)

# requirement 
pip install mitsuba
https://mitsuba.readthedocs.io/en/latest/
there might be some version issues

# examples
# python render.py -a -f ./samples/epn3d/plane --fov 16 -c '(73, 147, 240)'
# python render.py -a -f ./samples/epn3d/car --fov 24 -r 0.011 -c '(255, 135, 135)'
# python render.py -a -f ./samples/epn3d/chair  --fov 22 -r 0.01 -c '(175, 122, 179)'
# python render.py -a -f ./samples/epn3d/lamp  --fov 20  -r 0.009 -c '(132, 210, 197)'
# python render.py -a -f ./samples/epn3d/sofa --fov 25 -r 0.012  -c '(242, 146, 29)'
# python render.py -a -f ./samples/epn3d/table --fov 24 -r 0.011 -c '(147, 178, 98)'

# plane blue (73, 147, 240)
# car red (226, 104, 104)
# chair purple (175, 122, 179)
# lamp teal (132, 210, 197)
# sofa yellow (242, 146, 29)
# table green (147, 185, 98)
