"""
List of features - [X] = extractable: 
-----------------
1 , Static Probability,     Static probability of the net.
2,  Transition Probability, Activity from 0 to 1.
3,  [X] Controllability,        Controllability of the net.
4,  [X] Observability,          Observability of the net.

5,  [X] Fanin Level 1,          # of connected inputs at level 1
6,  [X] Fanout Level 1,         # of connected outputs at level 1
7,  [X] Fanin Level 2,          # of connected inputs at level 2
8,  [X] Fanout Level 2,         # of connected outputs at level 2
9,  [X] Nearest FF D,           Distance of the nearest flip-flop input
10, [X] Nearest FF Q,           Distance of the nearest flip-flop output
11, [X] Min. PI Distance,       Min. distance from nearest primary input
12, [X] Min. PO Distance,       Min. distance from nearest primary output
"""

gate_types = ("dff", "not", "and", "or", "nand", "nor")
primary_inputs = []
primary_outputs = []
netlist = {}
logic_gates = []


class Net: 
    def __init__(self, name, is_primary_output, is_primary_input):
        self.name = name
        self.is_primary_output = is_primary_output 
        self.is_primary_input = is_primary_input 
        self.fan_ins_lvl1 = []
        self.fan_ins_lvl2 = []
        self.fan_outs_lvl1 = []
        self.fan_outs_lvl2 = []
        self.cc0 = 0 
        self.cc1 = 0 
        self.observability = 0 
        self.nearest_ff_input = 0 
        self.nearest_ff_output = 0 
        self.min_pi_distance = 0 
        self.min_po_distance = 0 

    def find_fan_ins_level1(self): 
        for g in logic_gates: 
            if self.name in g.outputs: 
                for net_name in g.inputs: 
                    if net_name != "CK":  
                        self.fan_ins_lvl1.append((g.name, net_name))


        # print(self.name, [i for i in self.fan_ins_lvl1])

    def find_fan_outs_level1(self): 
        for g in logic_gates: 
            if self.name in g.inputs: 
                for net_name in g.outputs:
                    self.fan_outs_lvl1.append((g.name, net_name)) 

        # print(self.name, [i for i in self.fan_outs_lvl1])

    def find_fan_ins_level2(self): 
        for _, net_name in self.fan_ins_lvl1: 
            for lvl2_gate_name, lvl2_net_name in netlist[net_name].fan_ins_lvl1: 
                self.fan_ins_lvl2.append((lvl2_gate_name, lvl2_net_name))


        # print(self.name, [i for i in self.fan_ins_lvl2])


    def find_fan_outs_level2(self): 
        for _, net_name in self.fan_outs_lvl1: 
            for lvl2_gate_name, lvl2_net_name in netlist[net_name].fan_outs_lvl1: 
                self.fan_outs_lvl2.append((lvl2_gate_name, lvl2_net_name))


        # print(self.name, [i for i in self.fan_outs_lvl2])

    def find_controllability(self): 
        # Primary inputs
        if self.is_primary_input:
            self.cc0 = 1
            self.cc1 = 1
            return

        # Find the gate that drives THIS net
        driver = None
        for g in logic_gates:
            if self.name in g.outputs:
                driver = g
                break

        # No driver? Floating net.
        if driver is None:
            self.cc0 = 1
            self.cc1 = 1
            return

        gate_type = driver.name

        # Collect CC0/CC1 of input nets
        inputs = [netlist[n] for n in driver.inputs if n != "CK"]

        # ---------- GATE TYPES ----------
        if "NOT" in gate_type:
            In = inputs[0]
            self.cc0 = In.cc1 + 1
            self.cc1 = In.cc0 + 1

        elif "AND" in gate_type:
            a, b = inputs
            self.cc1 = a.cc1 + b.cc1 + 1
            self.cc0 = min(a.cc0, b.cc0) + 1

        elif "NAND" in gate_type:
            a, b = inputs
            # AND then NOT
            tmp_cc1 = a.cc1 + b.cc1 + 1
            tmp_cc0 = min(a.cc0, b.cc0) + 1
            self.cc1 = tmp_cc0 + 1
            self.cc0 = tmp_cc1 + 1

        elif "OR" in gate_type:
            a, b = inputs
            self.cc1 = min(a.cc1, b.cc1) + 1
            self.cc0 = a.cc0 + b.cc0 + 1

        elif "NOR" in gate_type: 
            a, b = inputs
            # OR then NOT
            tmp_cc1 = min(a.cc1, b.cc1) + 1
            tmp_cc0 = a.cc0 + b.cc0 + 1
            self.cc1 = tmp_cc0 + 1
            self.cc0 = tmp_cc1 + 1

        elif "DFF" in gate_type:
            Dnet = inputs[0]
            self.cc0 = Dnet.cc0 + 1
            self.cc1 = Dnet.cc1 + 1
        else:
            # default fallback
            self.cc0 = 1
            self.cc1 = 1

    def find_observability(self): 
        pass

    def find_nearest_ff_input(self): 
        pass

    def find_nearest_ff_output(self): 
        pass

    def find_min_pi_distance(self):
        pass

    def find_min_po_distance(self):
        pass



class LogicGate: 
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def add_inputs_outputs(self, gate_params):
        if len(gate_params) == 3:  # ck, dff, nor, and, or, etc 
            if gate_params[0] == "CK": # dff -> (clk, output, input)
                self.outputs.append(gate_params[1])  # add output 
                self.inputs.append(gate_params[0]) # ck is an input 
                self.inputs.append(gate_params[2]) # rest of inputs 
            else:  
                self.outputs.append(gate_params[0])  
                for param in gate_params[1:]: 
                    self.inputs.append(param)
        elif len(gate_params) == 2: # not (output, input)
            self.outputs.append(gate_params[0]) 
            self.inputs.append(gate_params[1])
    

# netlist_file_path = input("Enter netlist file you want to extract features from: ")
def parse(): 
    netlist_file_path = "s15850.v"
    with open(netlist_file_path, 'r') as f: 
        module_found = False 
        lines = f.readlines() 
        # parse 
        for line in lines: 
            if netlist_file_path.split('.')[0] in line:
                module_found = True 
                continue 

            if module_found:
                lsplt = line.strip().split()
                if lsplt: 
                    if lsplt[0] == "input": 
                        inputs = lsplt[1].split(',')
                        for i in inputs: 
                            if ';' in i:
                                i = i.replace(';', '')
                            netlist[i] = Net(i, True, False)
                    elif lsplt[0] == "output": 
                        outputs = lsplt[1].split(',')
                        for o in outputs: 
                            if ';' in o:
                                o = o.replace(';', '')
                            netlist[o] = Net(o, True, False)
                    elif lsplt[0] == "wire": 
                        wires = lsplt[1].split(',')
                        for w in wires: 
                            if ';' in w:
                                w = w.replace(';', '')
                            netlist[w] = Net(w, False, False)
                    elif lsplt[0] in gate_types:  # handle gates 
                        gate, params = lsplt[1].split('(')
                        nets = params.split(')')[0].split(',')
                        logic_gate = LogicGate(gate)
                        logic_gate.add_inputs_outputs(nets)
                        logic_gates.append(logic_gate)


def compute_all_controllability():
    # Initialize PIs and all FF outputs as controllable
    for n in netlist.values():
        if n.is_primary_input:
            n.cc0 = 1
            n.cc1 = 1

    # DFF outputs are pseudo-primary-inputs
    for g in logic_gates:
        if "DFF" in g.name:
            q = g.outputs[0]
            netlist[q].cc0 = 1
            netlist[q].cc1 = 1

    # Combinational logic
    changed = True
    while changed:
        changed = False
        for n in netlist.values():
            driver = None
            for g in logic_gates:
                if n.name in g.outputs:
                    driver = g
                    break

            if driver and "DFF" in driver.name:
                continue  

            old = (n.cc0, n.cc1)
            n.find_controllability()

            if (n.cc0, n.cc1) != old:
                changed = True


def propagate(): 
    for n in netlist: 
        netlist[n].find_fan_ins_level1()

    for n in netlist: 
        netlist[n].find_fan_ins_level2()

    for n in netlist: 
        netlist[n].find_fan_outs_level1()

    for n in netlist: 
        netlist[n].find_fan_outs_level2()

    compute_all_controllability()



if __name__ == "__main__": 
    parse()

    # debug 
    print("primary inputs", primary_inputs)
    print("primary outputs", primary_outputs)
    print("gates", [(l.name, l.inputs, l.outputs) for l in logic_gates])
    print("controllability", [(n, netlist[n].cc0, netlist[n].cc1) for n in netlist])
    print("netlist", netlist)
    propagate() 
