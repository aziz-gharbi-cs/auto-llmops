output "cluster_name" {
  value       = aws_eks_cluster.cluster.name
  description = "Name of the Kubernetes cluster"
}

output "cluster_endpoint" {
  value       = aws_eks_cluster.cluster.endpoint
  description = "Endpoint for Kubernetes API server"
}

output "cluster_certificate_authority_data" {
  value       = aws_eks_cluster.cluster.certificate_authority[0].data
  description = "Base64 encoded certificate data required to authenticate with the cluster"
}
