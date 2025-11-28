export interface InstanceType {
  id: string;
  created_at: string;
  owner_id: string;
  name: string;
  ip_address: string | null;
  port: number;
};