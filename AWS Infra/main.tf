# Backend
module "pre-backend" {
  source = "./backend/pre-setup"
}

module "backend" {
  source = "./backend"
}

#####################################################################################################
# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "mlops-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]  
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"] 
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"] 

  enable_nat_gateway = true
  single_nat_gateway = true  # Set to true for cost savings

  tags = {
    Terraform   = "true"
    Environment = "development"
  }
}
#####################################################################################################
#SG
module "mlflow-sg" {
  source = "terraform-aws-modules/security-group/aws"
  name   = "mlflow-sg"
  vpc_id = module.vpc.vpc_id

  ingress_cidr_blocks = ["0.0.0.0/0"]
  ingress_rules = ["https-443-tcp", "http-80-tcp", "postgresql-tcp", "ssh-tcp"]
  egress_rules  = ["all-all"]

  # Custom rules
  # ingress_with_cidr_blocks = [
  #   {
  #     from_port   = 8080
  #     to_port     = 8080
  #     protocol    = "tcp"
  #     description = "Jenkins port"
  #     cidr_blocks = "0.0.0.0/0"  
  #   },
  #   {
  #     from_port   = 9000
  #     to_port     = 9000
  #     protocol    = "tcp"
  #     description = "SonarQube port"
  #     cidr_blocks = "0.0.0.0/0"  
  #   },
  #   {
  #     from_port   = 9090
  #     to_port     = 9090  
  #     protocol    = "tcp"
  #     description = "Prometheus port"
  #     cidr_blocks = "0.0.0.0/0"  
  #   }
  # ]
}

module "monitoring-sg" {
  source = "terraform-aws-modules/security-group/aws"
  name   = "monitoring-sg"
  vpc_id = module.vpc.vpc_id

  ingress_cidr_blocks = ["0.0.0.0/0"]
  ingress_rules       = ["https-443-tcp", "http-80-tcp", "ssh-tcp"]
  egress_rules        = ["all-all"]

  ingress_with_cidr_blocks = [
    {
      from_port   = 3000
      to_port     = 3000
      protocol    = "tcp"
      description = "Grafana port"
      cidr_blocks = "0.0.0.0/0"  
    },
    {
      from_port   = 9090
      to_port     = 9090
      protocol    = "tcp"
      description = "Prometheus port"
      cidr_blocks = "0.0.0.0/0"  
    }
  ]
}


#####################################################################################################
# EC2
module "mlflow" {
  source              = "./modules/ec2"
  vpc_id              = module.vpc.vpc_id
  ami_id              = "ami-05778ef68e10b91d7"
  instance_type       = "t2.medium"
  key_name            = "instance-keypair"
  subnet_id           = module.vpc.public_subnets[0]
  security_group_id   = module.mlflow-sg.security_group_id
  instance_name       = "mlflow"
  volume_size         = 30
  #user_data_script    = "${path.root}/scriptfiles/jenkins_setup.sh"

  tags = {
    Environment = "dev"
    Role        = "Experiment Tracking"
  }
}


# module "sonarqube_server" {
#   source              = "./modules/ec2"
#   vpc_id              = module.vpc.vpc_id
#   ami_id              = "ami-05778ef68e10b91d7"
#   instance_type       = "t2.medium"
#   key_name            = "instance-keypair"
#   subnet_id           = module.vpc.public_subnets[0]
#   security_group_id   = module.sonarqube-sg.security_group_id
#   instance_name       = "sonarqube"
#   volume_size         = 30
#   user_data_script    = null

#   tags = {
#     Environment = "dev"
#     Role        = "QA"
#   }
# }

# module "database_server" {
#   source              = "./modules/ec2"
#   vpc_id              = module.vpc.vpc_id
#   ami_id              = "ami-05778ef68e10b91d7"
#   instance_type       = "t2.medium"
#   key_name            = "instance-keypair"
#   subnet_id           = module.vpc.public_subnets[0]
#   instance_name       = "database"
#   volume_size         = 30
#   user_data_script    = null
#   security_group_id   = module.jenkins-sg.security_group_id  # ✅ reuse SG nếu cần truy cập DB

#   tags = {
#     Environment = "dev"
#     Role        = "DB"
#   }
# }

module "monitoring_server" {
  source              = "./modules/ec2"
  vpc_id              = module.vpc.vpc_id
  ami_id              = "ami-05778ef68e10b91d7"
  instance_type       = "t2.medium"
  key_name            = "instance-keypair"
  subnet_id           = module.vpc.public_subnets[0]
  instance_name       = "monitoring"
  volume_size         = 80
  user_data_script    = "${path.root}/scriptfiles/prometheus_grafana.sh"
  security_group_id   = module.monitoring-sg.security_group_id

  tags = {
    Environment = "dev"
    Role        = "Monitoring"
  }
}





