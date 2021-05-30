
var nodes, edges, network;
var retrofitted_nodes, retrofitted_edges,retrofitted_network;

// convenience method to stringify a JSON object
function toJSON(obj) {
    return JSON.stringify(obj, null, 4);
}

function post(address,body,callback){
    var xhr = new XMLHttpRequest();
    console.log("Starting http req");
    xhr.open("POST", address, true);
    xhr.setRequestHeader('Accept', 'application/json; charset=UTF-8');
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');

    // send the collected data as JSON
    xhr.send(JSON.stringify(body));
    xhr.onreadystatechange = callback;
    console.log("done");
}

function get(address,body,callback){
    var xhr = new XMLHttpRequest();
    console.log("Starting http req");
    xhr.open("GET", address, true);
    // xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    // send the collected data as JSON
    xhr.send();
    xhr.onloadend = callback;
    console.log("done");
}


function findAndAddNeighborsOfNode() {
    var label = document.getElementById("node_name").value;
    var amount = document.getElementById("amount").value||10;
    try {
        //Empty out the previous.
        nodes.clear();
        retrofitted_nodes.clear();
        edges.clear();
        retrofitted_edges.clear();
        //Add initial node
        addNode(label,nodes);
        addNode(label,retrofitted_nodes);
        console.log("Starting the call");

        post('http://'+address+':4000/find_neighbors',{'input':label,'amount':amount},function () {
            if (this.readyState == 4) {
              if (this.status == 200) {
                  console.log("Responded");
                  var body = JSON.parse(this.responseText);//regular,post_specialized
                  let regs = body["regular"]
                  console.log(body);
                  let names = regs[0];
                  for (var i = 0; i < names.length; i++) {
                      addNode(regs[0][i],nodes);
                      addEdge(label, regs[0][i], regs[1][i],edges,nodes);
                  }
                  regs = body["post_specialized"]
                  console.log(body);
                  names = regs[1]
                  for (var i = 0; i < names.length; i++) {
                      addNode(regs[0][i],retrofitted_nodes);
                      addEdge(label, regs[0][i], regs[1][i],retrofitted_edges,retrofitted_nodes);
                  }
                  console.log("Finding and adding more relationships");
                  console.log("Done");
                  document.getElementById("loader").style.display = "none";

              }
            }
        });
    }
    catch (err) {
        alert(err);
    }
}
// function findAndAddRelationships(){
//     for (var i = 0; i <nodes.length; i++) {
//         for (var j = 0; j < nodes.length; j++) {
//             if(i==j){
//                 continue;
//             }
//             var body = {
//                 'start':nodes.get(i)["label"],
//                 'end':nodes.get(j)["label"]
//             };
//             post("http://localhost:4000/get_relations",body,function () {
//                 var body = JSON.parse(this.responseText);
//                 for (var k = 0; k < body["edges"].length; k++) {
//                     console.log(body["edges"][k]);
//                     var label = body["edges"][k]["rel"]["label"];
//                     var weight = body["edges"][k]["weight"];
//                     if(weight>10){
//                         weight = weight/10.0;
//                     }
//                     if(weight>1)
//                         weight=weight/10.0;
//                     if(weight==1){
//                         weight=weight*.75
//                     }
//                     var start = body["edges"][k]["start"]["term"];
//                     var end = body["edges"][k]["end"]["term"];
//                     var to_add_id = edges.length;
//                     // console.log(label);
//                     // console.log(window.color_dict)
//                     if (! Object.keys(window.color_dict).includes(label)){
//                         window.color_dict[label]=getRandomColor();
//                     }
//                     // console.log("colorict"+window.color_dict[label]);
//                     if(!(getEdgeIdValue(start,end,label)===null)){
//                         continue;
//                     }
//                     edges.add({
//                         id: to_add_id,
//                         from: getNodeIdValue(start),
//                         to: getNodeIdValue(end),
//                         label: label+" "+weight.toFixed(2),
//                         value: weight,
//                         smooth: {type: 'dynamic'},
//                         arrows: 'to',
//                         color:window.color_dict[label],
//                         arrowStrikethrough:false
//                     });
//                 }
//                 // console.log(body);
//             });
//         }
//     }
//     document.getElementById("loader").style.display = "none";
// }
function getRandomColor() {
    var max = 255;
    var min = 0;
    var colorstring = {color:'rgb('+
            Math.trunc( Math.random() * (max - min) + min)+','+
            Math.trunc( Math.random() * (max - min) + min)+','+
            Math.trunc( Math.random() * (max - min) + min)+')'};
    // console.log(colorstring);
    return colorstring;
}

function addEdge(from_node,to_node,value,edges,nodes) {
    try {
        var from_node_id = getNodeIdValue(from_node,nodes);
        var to_node_id = getNodeIdValue(to_node,nodes);
        var to_add_id = edges.length;
        edges.add({
            id: to_add_id,
            from: from_node_id,
            to: to_node_id,
            label:value.toString()
        });
    }
    catch (err) {
        alert(err);
    }
}


function getEdgeIdValue(start,end,label,edges,nodes){
    var result = null;
    var start_id = getNodeIdValue(start,nodes);
    var end_id = getNodeIdValue(end,nodes);
    for (var i = 0, len=edges.length; i < len; i++) {
        var edge = edges.get(i);
        if ((edge.label.substring(0,edge.label.indexOf(" "))===label&&edge.from===start_id&&edge.to===end_id)
            // ||
            // (edge.label===label&&edge.from===end_id&&edge.to===start_id)
        ){
            result = edge.id;
        }
    }
    return result;
}

function addNode(node_name,nodes) {
    var to_add_id = nodes.length;
    if(getNodeIdValue(node_name,nodes)===null){
        nodes.add({
            id: to_add_id,
            label: node_name
        });
    }
}

function getNodeIdValue(label,nodes){
    var result = null;
    for (var i = 0, len=nodes.length; i < len; i++) {
        var node = nodes.get(i);
        var repd = label.replace("/c/en/","");
        if (node.label===repd){
            result = node.id;
            break;
        }
    }
    return result;
}

function removeEdge(edge_id_value) {
    try {
        edges.remove({id: edge_id_value});
    }
    catch (err) {
        alert(err);
    }
}
function removeNode(node_id_value) {
    try {
        nodes.remove({id: node_id_value});
    }
    catch (err) {
        alert(err);
    }
}

function init_regular(containerName) {
    console.log("Initializing:"+containerName)
    // create an array with nodes
    nodes = new vis.DataSet();
    // nodes.on('*', function () {
    //   document.getElementById('nodes').innerHTML = JSON.stringify(nodes.get(), null, 4);
    // });
    // nodes.add([
    //     {id: '0', label: 'Node 1'},
    //     {id: '1', label: 'Node 2'},
    //     {id: '2', label: 'Node 3'},
    //     {id: '3', label: 'Node 4'},
    //     {id: '4', label: 'Node 5'}
    // ]);

    // create an array with edges
    edges = new vis.DataSet();
    // edges.on('*', function () {
    //   document.getElementById('edges').innerHTML = JSON.stringify(edges.get(), null, 4);
    // });
    // edges.add([
    //     {id: '0', from: '1', to: '2', value:'5',label:'Poop'},
    //     {id: '1', from: '1', to: '3', value:'1',label:'Poop1'},
    //     {id: '2', from: '2', to: '4', value:'2',label:'Poop2'},
    //     {id: '3', from: '2', to: '5', value:'6',label:'Poop3'}
    // ]);

    // create a network
    var container = document.getElementById(containerName);
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {
        // nodes:{
        //     scaling: {
        //         customScalingFunction: function (min, max, total, value) {
        //             return 10;
        //         }
        //     }
        // },
        edges:{
            scaling: {
                customScalingFunction: function (min, max, total, value) {
                    // if (value > 1) {
                    //     return value / 10
                    // } else
                    //     return value
                    var min = 0.1;
                    var sc = 1/total;
                    if(sc<min){
                        sc = min
                    }
                    return sc;
                }
            }
        },
        // layout: {
        //     hierarchical: {
        //         direction: 'UD'
        //     }
        // },
        physics: {
            repulsion: {
                nodeDistance: 100
            },
            minVelocity: 0.75,
            solver: "repulsion"
        }
    };

    network = new vis.Network(container, data, options);
    network.on("selectNode", function (params) {
        console.log('selectNode Event:', params);
        document.getElementById('selected_node').innerHTML = JSON.stringify(nodes.get(params["nodes"][0]))
    });
    network.on("selectEdge", function (params) {
        console.log('selectEdge Event:', params);
        document.getElementById('selected_edge').innerHTML = JSON.stringify(edges.get(params["edges"][0]))
    });
    network.on("deselectNode", function (params) {
        console.log('deselectNode Event:', params);
        document.getElementById('selected_node').innerHTML = "None"

    });
    network.on("deselectEdge", function (params) {
        console.log('deselectEdge Event:', params);
        document.getElementById('selected_edge').innerHTML = "None"

    });
    network.on("hoverNode", function (params) {
        console.log('hoverNode Event:', params);
    });
    network.on("hoverEdge", function (params) {
        console.log('hoverEdge Event:', params);
    });

    // updateAvailableModels(null);
}
function init_ps(containerName) {
    console.log("Initializing:"+containerName)
    // create an array with nodes
    retrofitted_nodes = new vis.DataSet();

    // create an array with edges
    retrofitted_edges = new vis.DataSet();

    // create a network
    var container = document.getElementById(containerName);
    var data = {
        nodes: retrofitted_nodes,
        edges: retrofitted_edges
    };
    var options = {

        edges:{
            scaling: {
                customScalingFunction: function (min, max, total, value) {
                    var min = 0.1;
                    var sc = 1/total;
                    if(sc<min){
                        sc = min
                    }
                    return sc;
                }
            }
        }
    };

    retrofitted_network = new vis.Network(container, data, options);
    retrofitted_network.on("selectNode", function (params) {
        console.log('selectNode Event:', params);
        document.getElementById('selected_node').innerHTML = JSON.stringify(retrofitted_nodes.get(params["nodes"][0]))
    });
    retrofitted_network.on("selectEdge", function (params) {
        console.log('selectEdge Event:', params);
        document.getElementById('selected_edge').innerHTML = JSON.stringify(retrofitted_edges.get(params["edges"][0]))
    });
    retrofitted_network.on("deselectNode", function (params) {
        console.log('deselectNode Event:', params);
        document.getElementById('selected_node').innerHTML = "None"

    });
    retrofitted_network.on("deselectEdge", function (params) {
        console.log('deselectEdge Event:', params);
        document.getElementById('selected_edge').innerHTML = "None"

    });
    retrofitted_network.on("hoverNode", function (params) {
        console.log('hoverNode Event:', params);
    });
    retrofitted_network.on("hoverEdge", function (params) {
        console.log('hoverEdge Event:', params);
    });

    // updateAvailableModels(null);
}

let address = "168.61.50.33";

init_regular('network');
init_ps('network2');