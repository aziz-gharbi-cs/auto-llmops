# Kubernetes Module - Cluster and Node Pools (Section 35 & 37)

resource "aws_eks_cluster" "cluster" {
  name     = "${var.environment}-llmops-cluster"
  role_arn = var.cluster_role_arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids = var.subnet_ids
  }

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Standard CPU Node Group for API and Gateway
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.cluster.name
  node_group_name = "${var.environment}-cpu-nodes"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.cpu_instance_types

  scaling_config {
    desired_size = var.cpu_desired_size
    min_size     = var.cpu_min_size
    max_size     = var.cpu_max_size
  }

  labels = {
    role = "general-api"
  }
}

# GPU Node Group for Model Inference (vLLM / Gemma 3)
resource "aws_eks_node_group" "gpu_nodes" {
  count           = var.enable_gpu_nodes ? 1 : 0
  cluster_name    = aws_eks_cluster.cluster.name
  node_group_name = "${var.environment}-gpu-nodes"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.subnet_ids
  instance_types  = var.gpu_instance_types

  scaling_config {
    desired_size = var.gpu_desired_size
    min_size     = var.gpu_min_size
    max_size     = var.gpu_max_size
  }

  labels = {
    role        = "model-inference"
    accelerator = "nvidia-gpu"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}
