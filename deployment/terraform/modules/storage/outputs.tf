output "bucket_id" {
  value       = aws_s3_bucket.artifacts.id
  description = "Name of the storage bucket"
}

output "bucket_arn" {
  value       = aws_s3_bucket.artifacts.arn
  description = "ARN of the storage bucket"
}
