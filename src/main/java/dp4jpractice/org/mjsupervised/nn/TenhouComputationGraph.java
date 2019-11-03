package dp4jpractice.org.mjsupervised.nn;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class TenhouComputationGraph extends ComputationGraph {
    private Logger logger = LoggerFactory.getLogger(TenhouComputationGraph.class);

    public TenhouComputationGraph(ComputationGraphConfiguration configuration) {
        super(configuration);
    }

    @Override
    public void computeGradientAndScore() {
        if (configuration.getBackpropType() == BackpropType.TruncatedBPTT) {
            logger.debug("++++++++++++++++++++++++++++++++ " + "Truncated");
        }else {
            logger.debug("++++++++++++++++++++++++++++++++ " + "Standard");
        }

        super.computeGradientAndScore();
    }

    @Override
    public TenhouComputationGraph clone() {
        System.out.println("----------------------------------> TenhouComputationGraph clone");
        TenhouComputationGraph cg = new TenhouComputationGraph(configuration.clone());
        cg.init(params().dup(), false);
        if (solver != null) {
            //If  solver is null: updater hasn't been initialized -> getUpdater call will force initialization, however
            ComputationGraphUpdater u = this.getUpdater();
            INDArray updaterState = u.getStateViewArray();
            if (updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState.dup());
            }
        }
        return cg;
    }


    private INDArray reshapeTimeStepInput(INDArray input) {
        if (input.rank() == 2) { // dynamically reshape to 3D input with one time-step.
            long[] inShape = input.shape();
            input = input.reshape(inShape[0], inShape[1], 1);
        }
        return input;
    }

    @Override
    protected INDArray[] outputOfLayersDetached(boolean train, FwdPassType fwdPassType, int[] layerIndexes, INDArray[] features,
                                                INDArray[] fMask, INDArray[] lMasks, boolean clearLayerInputs, boolean detachedInputs, MemoryWorkspace outputWorkspace){
        int numInputArrays = 1;
        if(features.length != numInputArrays){
            throw new IllegalArgumentException("Invalid number of input arrays: network has " + numInputArrays
                    + " inputs, got " + features.length + " input arrays");
        }
        for( int i=0; i<layerIndexes.length; i++ ) {
            if(layerIndexes[i] < 0 || layerIndexes[i] >= topologicalOrder.length) {
                throw new IllegalArgumentException("Invalid input index - index must be >= 0 and < " + topologicalOrder.length
                        + ", got index " + layerIndexes[i]);
            }
        }
        setInputs(features);
        setLayerMaskArrays(fMask, lMasks);

        MemoryWorkspace outputPrevious = null;
        if(outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
            //Verify that no workspace is open externally
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active before call to outputOfLayersDetached");
        } else {
            Preconditions.checkState(outputWorkspace.isScopeActive(), "Workspace \"" + outputWorkspace.getId() +
                    "\" was provided for the network/layer outputs. When provided, this workspace must be opened before " +
                    "calling the output method; furthermore, closing the workspace is the responsibility of the user");
            outputPrevious = outputWorkspace.getParentWorkspace();
        }


        //First: for each vertex, determine the highest index of the vertex that consumes it's output
        //Then: for each vertex, determine the forward pass step that each vertex's output has been fully consumed on
        //In other words, if vertex X -> Y and X -> Z, and topological sort order is X,a,Y,b,Z,
        //Then we know X's output activations have been fully consumed by step index 4 in the topological sort
        //thus vertexOutputsFullyConsumedByStep[X.index] = IndexOf(topologicalSort, Z.index)

        //Position in array: index of vertex. Value at position: the step (in topological order) that the activations
        // have been consumed by
        //Put another way: this is the step that it's safe to deallocate the layer's activations by closing the
        // corresponding workspace
        int[] vertexOutputsFullyConsumedByStep = new int[topologicalOrder.length];
        for(GraphVertex gv : vertices){
            int idx = gv.getVertexIndex();
            int maxStepOfOutputTo = -1;
            VertexIndices[] outputsTo = gv.getOutputVertices();
            if(outputsTo != null) {
                //May be null for final/output layers
                for (VertexIndices vi : outputsTo) {
                    int posInTopoSort = ArrayUtils.indexOf(topologicalOrder, vi.getVertexIndex());
                    if (posInTopoSort == -1) {
                        throw new IllegalStateException("Did not find vertex " + vi.getVertexIndex() + " in topological sort array");
                    }
                    maxStepOfOutputTo = Math.max(maxStepOfOutputTo, posInTopoSort);
                }
            } else {
                maxStepOfOutputTo = topologicalOrder.length-1;
            }
            vertexOutputsFullyConsumedByStep[idx] = maxStepOfOutputTo;
        }

        //Do forward pass according to the topological ordering of the network
        INDArray[] outputs = new INDArray[layerIndexes.length];
        int stopIndex = -1;
        for( int i=0; i<layerIndexes.length; i++ ){
            stopIndex = Math.max(stopIndex, ArrayUtils.indexOf(topologicalOrder, layerIndexes[i]));
        }
        List<LayerWorkspaceMgr> allWorkspaceManagers = new ArrayList<LayerWorkspaceMgr>();
        List<LayerWorkspaceMgr> freeWorkspaceManagers = new ArrayList<LayerWorkspaceMgr>();  //Basically used as a stack
        Map<MemoryWorkspace, LayerWorkspaceMgr> openActivationsWorkspaces = new IdentityHashMap<MemoryWorkspace, LayerWorkspaceMgr>();

        WorkspaceMode wsm = (train ? configuration.getTrainingWorkspaceMode() : configuration.getInferenceWorkspaceMode());
        boolean noWS = wsm == WorkspaceMode.NONE;
        LayerWorkspaceMgr allNone = noWS ? LayerWorkspaceMgr.noWorkspaces(helperWorkspaces) : null;
        List<MemoryWorkspace>[] closeAtEndIteraton = (List<MemoryWorkspace>[])new List[topologicalOrder.length];
        MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Throwable t = null;
        try {
            for (int i = 0; i <= stopIndex; i++) {
                GraphVertex current = vertices[topologicalOrder[i]];
                String vName = current.getVertexName();
                int vIdx = current.getVertexIndex();

                //First: determine what workspace manager we should use for forward pass in this vertex
                LayerWorkspaceMgr workspaceMgr;
                if (noWS) {
                    workspaceMgr = allNone;
                } else {
                    //First: is there a free forward pass workspace we can use?
                    if (freeWorkspaceManagers.size() > 0) {
                        workspaceMgr = freeWorkspaceManagers.remove(freeWorkspaceManagers.size() - 1);
                    } else {
                        //No existing free workspace managers for forward pass - create a new one...
                        String wsName = "WS_LAYER_ACT_" + allWorkspaceManagers.size();
                        workspaceMgr = LayerWorkspaceMgr.builder()
                                .with(ArrayType.INPUT, wsName, WS_LAYER_ACT_X_CONFIG)
                                .with(ArrayType.ACTIVATIONS, wsName, WS_LAYER_ACT_X_CONFIG)
                                .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                                .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                                .build();

                        if (detachedInputs) {
                            //Sometimes (like: external errors use cases) we don't want the activations/inputs to be
                            // in a workspace
                            workspaceMgr.setScopedOutFor(ArrayType.INPUT);
                            workspaceMgr.setScopedOutFor(ArrayType.ACTIVATIONS);
                        } else {
                            //Don't leverage out of async MultiDataSetIterator workspaces
                            if (features[0].isAttached()) {
                                workspaceMgr.setNoLeverageOverride(features[0].data().getParentWorkspace().getId());
                            }
                        }

                        allWorkspaceManagers.add(workspaceMgr);
                    }
                }
                workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

                //Is this one of the layers/vertices that we want the output for?
                boolean isRequiredOutput = false;
                String origWSAct = null;
                WorkspaceConfiguration origWSActConf = null;
                if (ArrayUtils.contains(layerIndexes, vIdx)) {
                    isRequiredOutput = true;

                    if (outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)) {
                        //Place activations in user-specified workspace
                        origWSAct = workspaceMgr.getWorkspaceName(ArrayType.ACTIVATIONS);
                        origWSActConf = workspaceMgr.getConfiguration(ArrayType.ACTIVATIONS);
                        workspaceMgr.setWorkspace(ArrayType.ACTIVATIONS, outputWorkspace.getId(), outputWorkspace.getWorkspaceConfiguration());
                    } else {
                        //Standard case
                        if (!workspaceMgr.isScopedOut(ArrayType.ACTIVATIONS)) {
                            //Activations/output to return: don't want this in any workspace
                            origWSAct = workspaceMgr.getWorkspaceName(ArrayType.ACTIVATIONS);
                            origWSActConf = workspaceMgr.getConfiguration(ArrayType.ACTIVATIONS);
                            workspaceMgr.setScopedOutFor(ArrayType.ACTIVATIONS);
                        }
                    }
                }

                //Open the relevant workspace for the activations.
                //Note that this will be closed only once the current vertex's activations have been consumed
                MemoryWorkspace wsActivations = null;
                if (outputWorkspace == null || outputWorkspace instanceof DummyWorkspace || !isRequiredOutput) {    //Open WS if (a) no external/output WS (if present, it's already open), or (b) not being placed in external/output WS
                    wsActivations = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS);
                    openActivationsWorkspaces.put(wsActivations, workspaceMgr);
                }

                //Note that because we're opening activation workspaces not in any defined order (i.e., workspace
                // use isn't simply nested), we'll manually override the previous workspace setting. Otherwise, when we
                // close these workspaces, the "current" workspace may be set to the incorrect one
                if (wsActivations != null)
                    wsActivations.setPreviousWorkspace(initialWorkspace);

                int closeableAt = vertexOutputsFullyConsumedByStep[vIdx];
                if (outputWorkspace == null || outputWorkspace instanceof DummyWorkspace || (wsActivations != null && !outputWorkspace.getId().equals(wsActivations.getId()))) {
                    if (closeAtEndIteraton[closeableAt] == null) {
                        closeAtEndIteraton[closeableAt] = new ArrayList<MemoryWorkspace>();
                    }
                    closeAtEndIteraton[closeableAt].add(wsActivations);
                }


                try (MemoryWorkspace wsFFWorking = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {
                    VertexIndices[] inputsTo = current.getOutputVertices();
                    logger.debug("Deal with one layer ");

                    INDArray out;
                    if (current.isInputVertex()) {
                        logger.debug("Input vertex");
                        out = features[vIdx];
                    } else {
                        logger.debug("Internal vertex " + current.getVertexName());
                        if (fwdPassType == FwdPassType.STANDARD) {
                            //Standard feed-forward case
                            logger.debug("Standard ff");
                            out = current.doForward(train, workspaceMgr);
                        } else if (fwdPassType == FwdPassType.RNN_TIMESTEP) {
                            logger.debug("RNN_Timestep");
                            if (current.hasLayer()) {
                                //Layer
                                logger.debug("Has layer");
                                INDArray input = current.getInputs()[0];
                                Layer l = current.getLayer();
                                if (l instanceof RecurrentLayer) {
                                    logger.debug("Recurrent layer");
                                    out = ((RecurrentLayer) l).rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                                } else if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer && ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying() instanceof RecurrentLayer) {
                                    logger.debug("Wrapper recurrent layer");
                                    RecurrentLayer rl = ((RecurrentLayer) ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying());
                                    out = rl.rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                                } else if (l instanceof MultiLayerNetwork) {
                                    logger.debug("Another network");
                                    out = ((MultiLayerNetwork) l).rnnTimeStep(reshapeTimeStepInput(input));
                                } else {
                                    //non-recurrent layer
                                    logger.debug("Not recurrent layer");
                                    out = current.doForward(train, workspaceMgr);
                                }
                            } else {
                                logger.debug("Another graph node");
                                //GraphNode
                                out = current.doForward(train, workspaceMgr);
                            }
                        } else {
                            throw new IllegalArgumentException("Unsupported forward pass type for this method: " + fwdPassType);
                        }
                        validateArrayWorkspaces(workspaceMgr, out, ArrayType.ACTIVATIONS, vName, false, "Feed forward (inference)");
                    }

                    if (inputsTo != null) {  //Output vertices may not input to any other vertices
                        for (VertexIndices v : inputsTo) {
                            //Note that we don't have to do anything special here: the activations are always detached in
                            // this method
                            int inputToIndex = v.getVertexIndex();
                            int vIdxEdge = v.getVertexEdgeNumber();
                            logger.debug("To setInput for vertice " + vertices[inputToIndex].toString());
                            long[] outShape = out.shape();
                            for (long l : outShape) {
                                logger.debug("shape " + l);
                            }
//                            logger.debug("The output " + out.toString());
//                            INDArray verticeInputs = vertices[inputToIndex].getInputs()[0];
//                            logger.debug("Input " );
//                            logger.debug(verticeInputs.toStringFull());

//                            vertices[inputToIndex]
                            logger.debug("Batch size " + this.batchSize());
                            vertices[inputToIndex].setInput(vIdxEdge, out, workspaceMgr);
                        }
                    }

                    if (clearLayerInputs) {
                        current.clear();
                    }

                    if (isRequiredOutput) {
                        outputs[ArrayUtils.indexOf(layerIndexes, vIdx)] = out;
                        if (origWSAct != null) {
                            //Reset the configuration, as we may reuse this workspace manager...
                            workspaceMgr.setWorkspace(ArrayType.ACTIVATIONS, origWSAct, origWSActConf);
                        }
                    }
                }

                //Close any activations workspaces that we no longer require
                //Note that activations workspaces can be closed only once the corresponding output activations have
                // been fully consumed
                if (closeAtEndIteraton[i] != null) {
                    for (MemoryWorkspace wsAct : closeAtEndIteraton[i]) {
                        wsAct.close();
                        LayerWorkspaceMgr canNowReuse = openActivationsWorkspaces.remove(wsAct);
                        freeWorkspaceManagers.add(canNowReuse);
                    }
                }
            }
        } catch (Throwable t2){
            t = t2;
        } finally {
            //Close all open workspaces... usually this list will be empty, but not if an exception is thrown
            //Though if stopIndex < numLayers, some might still be open
            for(MemoryWorkspace ws : openActivationsWorkspaces.keySet()){
                while (ws.isScopeActive()) {
                    //Edge case here: seems that scoping out can increase the tagScope of the current WS
                    //and if we hit an exception during forward pass, we aren't guaranteed to call close a sufficient
                    // number of times to actually close it, in all cases
                    try{
                        ws.close();
                    } catch (Throwable t2){
                        if(t != null){
                            logger.error("Encountered second exception while trying to close workspace after initial exception");
                            logger.error("Original exception:", t);
                            throw t2;
//                            t2.printStackTrace();
                        }
                    }
                }
            }
            Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);

            if(t != null){
                if(t instanceof RuntimeException){
                    throw ((RuntimeException)t);
                }
                throw new RuntimeException("Error during neural network forward pass", t);
            }

            if(outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
                WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active at the end of outputOfLayerDetached");
            } else {
                Preconditions.checkState(outputWorkspace.isScopeActive(), "Expected output workspace to still be open" +
                        "at end of outputOfLayerDetached, but ");
                outputWorkspace.setPreviousWorkspace(outputPrevious);
            }
        }

        return outputs;
    }
}
