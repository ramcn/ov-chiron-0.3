from openvino.inference_engine import IENetwork, IEPlugin
import argparse,os,time,sys

sys.path.append('/opt/intel/openvino/python/python2.7/openvino/inference_engine/')

model_xml = "/home/guest-intern/new-model/Chiron-0.3/frozen_inference.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin"
plugin = IEPlugin(device="CPU", plugin_dirs="/opt/intel/openvino/inference_engine/lib/intel64")
#plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx512.so")
# Read IR
print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = IENetwork(model=model_xml, weights=model_bin)
supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
       print("Following layers are not supported by the plugin for specified device {}:\n {}".
                   format(plugin.device, ', '.join(not_supported_layers)))
       sys.exit(1)
print("Num input: ")
print(net.inputs.keys())
print("Num output: ")
print(net.outputs)
assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
assert len(net.outputs) == 1, "Sample supports only single output topologies"
print('Loading IR to the plugin...')
exec_net = plugin.load(network=net, num_requests=1)
